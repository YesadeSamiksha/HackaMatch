import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_NAME, APP_TAGLINE, API_VERSION, GEOGRAPHIC_SCOPE, PHASE
from app.dataset_registry import DatasetRegistry
from app.loaders.challan_loader import ChallanLoader
from app.loaders.traffic_csv_loader import TrafficCsvLoader
from app.loaders.rainfall_loader import RainfallLoader
from app.loaders.registration_loader import RegistrationLoader
from app.services.overview_service import OverviewService
from app.services.quality_service import QualityService
from app.services.insights_service import InsightsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("citypulse")

app = FastAPI(
    title=f"{APP_NAME} API — {APP_TAGLINE}",
    description="Mumbai-focused traffic analytics and road-safety decision-support backend for CityPulse (Phase 1).",
    version=API_VERSION
)

# CORS middleware allowing frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons
registry = DatasetRegistry()
challan_loader = ChallanLoader()
traffic_loader = TrafficCsvLoader()
rainfall_loader = RainfallLoader()
registration_loader = RegistrationLoader()
overview_service = OverviewService()
quality_service = QualityService()
insights_service = InsightsService()


@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "app_name": APP_NAME,
        "product_name": APP_TAGLINE,
        "geographic_scope": GEOGRAPHIC_SCOPE,
        "phase": PHASE,
        "version": API_VERSION
    }


@app.get("/api/datasets")
def get_datasets():
    datasets = registry.get_all_datasets()
    return {
        "geographic_scope": GEOGRAPHIC_SCOPE,
        "total_datasets": len(datasets),
        "datasets": datasets
    }


@app.get("/api/datasets/{dataset_id}/metadata")
def get_dataset_metadata(dataset_id: str):
    entry = registry.get_dataset_by_id(dataset_id)
    if entry.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return entry


@app.get("/api/datasets/{dataset_id}/quality")
def get_dataset_quality(dataset_id: str):
    quality_report = quality_service.get_quality_report()
    report = next((r for r in quality_report["reports"] if r["dataset_id"] == dataset_id), None)
    if not report:
        raise HTTPException(status_code=404, detail=f"Quality report for dataset '{dataset_id}' not found")
    return report


@app.get("/api/quality")
def get_full_quality_report():
    return quality_service.get_quality_report()


@app.get("/api/overview")
def get_overview():
    return overview_service.get_overview()


@app.get("/api/mumbai/challans")
def get_mumbai_challans(
    type: str = Query("summary", description="divisions, police_stations, categories, vehicles, drunk_drive, or summary"),
    search: Optional[str] = Query(None, description="Search term for division or police station name"),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    order: str = Query("desc", description="asc or desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    data = challan_loader.load()
    if data.get("status") == "missing_file":
        raise HTTPException(status_code=404, detail=data.get("error"))

    if type == "summary":
        return {
            "metadata": data.get("metadata"),
            "summary": data.get("summary")
        }

    elif type == "divisions":
        div_data = data.get("division_challans", {})
        items = list(div_data.get("items", []))
        if search:
            items = [item for item in items if search.lower() in item["division"].lower()]
        
        # Sort
        if sort_by and items and sort_by in items[0]:
            items.sort(key=lambda x: x.get(sort_by, 0), reverse=(order == "desc"))
        else:
            items.sort(key=lambda x: x.get("total_challan", 0), reverse=(order == "desc"))

        total_items = len(items)
        start_idx = (page - 1) * page_size
        paged_items = items[start_idx : start_idx + page_size]

        return {
            "type": "divisions",
            "metadata": data.get("metadata"),
            "grand_total": div_data.get("grand_total"),
            "total_items": total_items,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_items + page_size - 1) // page_size if page_size > 0 else 1,
            "items": paged_items
        }

    elif type == "police_stations":
        ps_data = data.get("police_station_challans", {})
        items = list(ps_data.get("items", []))
        if search:
            items = [item for item in items if search.lower() in item["police_station"].lower()]

        if sort_by and items and sort_by in items[0]:
            items.sort(key=lambda x: x.get(sort_by, 0), reverse=(order == "desc"))
        else:
            items.sort(key=lambda x: x.get("total_challan", 0), reverse=(order == "desc"))

        total_items = len(items)
        start_idx = (page - 1) * page_size
        paged_items = items[start_idx : start_idx + page_size]

        return {
            "type": "police_stations",
            "metadata": data.get("metadata"),
            "grand_total": ps_data.get("grand_total"),
            "total_items": total_items,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_items + page_size - 1) // page_size if page_size > 0 else 1,
            "items": paged_items
        }

    elif type == "categories":
        cat_data = data.get("violation_categories", {})
        items = cat_data.get("category_totals", [])
        if search:
            items = [item for item in items if search.lower() in item["category"].lower()]
        return {
            "type": "categories",
            "metadata": data.get("metadata"),
            "grand_total": cat_data.get("grand_total"),
            "items": items,
            "division_breakdown": cat_data.get("division_breakdown", [])
        }

    elif type == "vehicles":
        veh_data = data.get("vehicle_type_challans", {})
        items = veh_data.get("items", [])
        if search:
            items = [item for item in items if search.lower() in item["vehicle_type"].lower()]
        return {
            "type": "vehicles",
            "metadata": data.get("metadata"),
            "grand_total": veh_data.get("grand_total"),
            "items": items
        }

    elif type == "drunk_drive":
        dd_data = data.get("drunk_drive_challans", {})
        return {
            "type": "drunk_drive",
            "metadata": data.get("metadata"),
            "division_wise": dd_data.get("division_wise", {}),
            "police_station_wise": dd_data.get("police_station_wise", {})
        }

    else:
        raise HTTPException(status_code=400, detail=f"Invalid type '{type}'. Use divisions, police_stations, categories, vehicles, drunk_drive, or summary")


@app.get("/api/mumbai/traffic-preview")
def get_mumbai_traffic_preview(
    area: Optional[str] = Query(None, description="Filter by area e.g. Bandra, Malad, Andheri, Dadar, Lower Parel"),
    vehicle_type: Optional[str] = Query(None, description="Filter by vehicle type e.g. Auto, Bus, Truck, Car, Bike"),
    congestion_level: Optional[str] = Query(None, description="Filter by congestion level e.g. Low, Medium, High"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    base_data = traffic_loader.load()
    if base_data.get("status") == "missing_file":
        raise HTTPException(status_code=404, detail=base_data.get("error"))

    records_result = traffic_loader.get_filtered_records(
        area=area,
        vehicle_type=vehicle_type,
        congestion_level=congestion_level,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size
    )

    return {
        "status": base_data.get("status"),
        "metadata": base_data.get("metadata"),
        "columns_info": base_data.get("columns_info"),
        "filter_options": base_data.get("filter_options"),
        "summary_statistics": base_data.get("summary_statistics"),
        "distributions": base_data.get("distributions"),
        "pagination": {
            "total_records": records_result["total_records"],
            "page": records_result["page"],
            "page_size": records_result["page_size"],
            "total_pages": records_result["total_pages"]
        },
        "records": records_result["records"]
    }


@app.get("/api/mumbai/rainfall")
def get_mumbai_rainfall():
    data = rainfall_loader.load()
    if data.get("status") == "missing_file":
        raise HTTPException(status_code=404, detail=data.get("error"))
    return data


@app.get("/api/mumbai/vehicle-registrations")
def get_vehicle_registrations():
    reg_data = registration_loader.load()
    # Also load official Mumbai vehicle-type challans for contrasting authorized vs unverified data
    challan_data = challan_loader.load()
    mumbai_veh_challans = challan_data.get("vehicle_type_challans", {})

    return {
        "geographic_suitability": "Not suitable for Mumbai-only analysis",
        "has_mumbai_scope": False,
        "disclaimer": "Vehicle registrations are not the same as real-time road traffic counts. Excluded from Mumbai-specific metrics due to missing city identifier.",
        "unverified_registration_data": reg_data,
        "official_mumbai_enforcement_by_vehicle": {
            "source": "Mumbai Traffic Police 2024 Official Report (Page 8)",
            "grand_total": mumbai_veh_challans.get("grand_total", {}),
            "items": mumbai_veh_challans.get("items", [])
        }
    }


@app.get("/api/insights")
def get_insights():
    return insights_service.get_insights()


@app.get("/api/ml-lab/modules")
def get_ml_lab_modules():
    return {
        "disclaimer": "Machine-learning training has not started. Mumbai-specific modelling will begin only when suitable, verified Mumbai traffic observations and a clearly defined target variable are available.",
        "phase": "Phase 1: Architecture Skeleton & Data Readiness (No Trained ML Models)",
        "readiness_criteria": [
            {
                "criterion": "High-Frequency Physical Sensors",
                "status": "Missing",
                "detail": "Need calibrated physical road sensors (loop detectors, automated camera traffic counts) for Mumbai road segments."
            },
            {
                "criterion": "Continuous Granular Timestamps",
                "status": "Pending",
                "detail": "Current official PDF contains annual totals; ML forecasting requires 5-minute to 15-minute sensor intervals."
            },
            {
                "criterion": "Target Variable Ground Truth",
                "status": "Defined (Future)",
                "detail": "Target variable will be Segment Average Speed (km/h) or Level of Service (LOS A-F)."
            },
            {
                "criterion": "Spatiotemporal Adjacency Matrix",
                "status": "In Preparation",
                "detail": "Mumbai road network topology graph (excluding US PeMS sensor layouts) to be generated from OpenStreetMap."
            }
        ],
        "planned_modules": [
            {
                "id": "speed_prediction",
                "title": "Arterial Speed Prediction (Spatio-Temporal GNN)",
                "status": "Planned",
                "description": "Predict segment-level average vehicular speed 15, 30, and 60 minutes ahead across key corridors (e.g. Western Express Highway, Eastern Freeway).",
                "target_variable": "Segment Average Speed (km/h)",
                "input_features": ["Lagged speed (t-1...t-12)", "Historical speed profile", "Day of week", "Monsoon rainfall indicator", "Special events / VIP movement"],
                "split_plan": "Chronological split (70% train, 15% validation, 15% test) without temporal leakage",
                "baseline_model": "Historical Average & Vector Autoregression (VAR)",
                "candidate_models": ["DCRNN (Diffusion Convolutional RNN)", "Spatial-Temporal Graph Convolutional Network (STGCN)", "LightGBM regressor with spatial lags"],
                "evaluation_metrics": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Percentage Error (MAPE)"],
                "limitations": "Model cannot generalize to unmonitored secondary roads; performance subject to sensor dropouts during intense monsoon storms."
            },
            {
                "id": "congestion_classification",
                "title": "Junction Congestion Severity Classifier",
                "status": "Planned",
                "description": "Multi-class classification of junction bottleneck states into standardized Level of Service tiers (Free Flow, Moderate, Severe, Breakdown).",
                "target_variable": "Level of Service Tier (LOS Category)",
                "input_features": ["Queue length estimate", "Junction approach speed", "Signal cycle duration", "Rainfall intensity (mm/h)"],
                "split_plan": "Stratified temporal splits across monsoon and non-monsoon test windows",
                "baseline_model": "Rule-based thresholding & Logistic Regression",
                "candidate_models": ["XGBoost Classifier", "Random Forest Ensemble", "Graph Attention Network (GAT)"],
                "evaluation_metrics": ["Macro F1-Score", "Balanced Accuracy", "Confusion Matrix per LOS Class"],
                "limitations": "High imbalance in 'Breakdown' state requires focal loss or oversampling techniques."
            },
            {
                "id": "explainable_ai",
                "title": "Explainable AI (XAI) Feature Attribution Hub",
                "status": "Planned",
                "description": "SHAP (SHapley Additive exPlanations) and TreeSHAP attribution to explain why a particular corridor is forecasted to enter severe congestion.",
                "target_variable": "Feature Contribution Delta (Δ km/h)",
                "input_features": ["All candidate model features"],
                "split_plan": "Applied across holdout test set anomalies",
                "baseline_model": "Linear regression coefficients",
                "candidate_models": ["TreeSHAP", "KernelSHAP", "Integrated Gradients for GNNs"],
                "evaluation_metrics": ["Attribution stability", "Monotonicity score", "Expert interpretability validation"],
                "limitations": "Attribution indicates correlational feature importance, not counterfactual physical causation."
            },
            {
                "id": "model_evaluation_benchmarking",
                "title": "Model Evaluation & Drift Monitoring",
                "status": "Planned",
                "description": "Continuous monitoring of concept drift, seasonal monsoon shifts, and post-enforcement behavioural changes.",
                "target_variable": "Model Residual Tracking (Error distribution)",
                "input_features": ["Prediction residuals over rolling 7-day windows"],
                "split_plan": "Rolling out-of-time evaluation",
                "baseline_model": "Fixed validation baseline",
                "candidate_models": ["Evidently AI / Population Stability Index (PSI)", "CUSUM change-point detection"],
                "evaluation_metrics": ["Wasserstein Distance", "Kolmogorov-Smirnov Test p-value"],
                "limitations": "Requires automated continuous sensor ingestion pipeline."
            }
        ]
    }
