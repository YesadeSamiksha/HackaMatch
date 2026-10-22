import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Mumbai City" in data["geographic_scope"]
    assert "CityPulse" in data["app_name"]

def test_datasets_registry():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["total_datasets"] == 5
    ids = [d["id"] for d in data["datasets"]]
    assert "mumbai_traffic_challans_2024" in ids
    assert "maharashtra_traffic_csv" in ids
    assert "imd_mumbai_rainfall" in ids
    assert "vehicle_registrations_summary" in ids
    assert "metr_la_pems_benchmark" in ids

def test_overview():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["geographic_scope"] == "Mumbai City, Maharashtra, India"
    assert len(data["kpis"]) == 6
    assert len(data["top_enforcement_divisions"]) == 5
    assert len(data["top_violation_categories"]) == 5

def test_mumbai_challans_divisions():
    response = client.get("/api/mumbai/challans?type=divisions&page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "divisions"
    assert data["total_items"] == 42
    assert len(data["items"]) == 10
    # Grand total check
    assert data["grand_total"]["total_challan"] == 5937151

def test_mumbai_challans_police_stations():
    response = client.get("/api/mumbai/challans?type=police_stations&page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "police_stations"
    assert data["total_items"] == 91
    assert data["grand_total"]["total_challan"] == 847766

def test_mumbai_challans_categories():
    response = client.get("/api/mumbai/challans?type=categories")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "categories"
    assert len(data["items"]) == 24
    assert data["grand_total"] == 5937151

def test_traffic_preview_mumbai_filter():
    response = client.get("/api/mumbai/traffic-preview")
    assert response.status_code == 200
    data = response.json()
    meta = data["metadata"]
    # Check strict row filtering
    assert meta["total_rows_raw"] == 1200
    assert meta["mumbai_rows_retained"] == 234
    assert meta["non_mumbai_rows_excluded"] == 966
    # Ensure all returned records are Mumbai
    for r in data["records"]:
        assert r["City"].lower() == "mumbai"

def test_rainfall_mumbai_city_only():
    response = client.get("/api/mumbai/rainfall")
    assert response.status_code == 200
    data = response.json()
    meta = data["metadata"]
    assert "Mumbai City" in meta["geographic_scope"]
    assert len(data["records"]) == 65
    assert data["summary"]["observation_count"] == 65

def test_vehicle_registrations_not_suitable_for_mumbai():
    response = client.get("/api/mumbai/vehicle-registrations")
    assert response.status_code == 200
    data = response.json()
    assert data["geographic_suitability"] == "Not suitable for Mumbai-only analysis"
    assert data["has_mumbai_scope"] is False
    assert "Vehicle registrations are not the same as real-time road traffic counts" in data["disclaimer"]

def test_insights():
    response = client.get("/api/insights")
    assert response.status_code == 200
    data = response.json()
    assert data["insights_count"] >= 5
    for item in data["items"]:
        assert "observed" in item
        assert "interpretation" in item
        assert "suggested_next_step" in item
        assert "can_infer" in item
        assert "cannot_infer" in item

def test_ml_lab_no_trained_models():
    response = client.get("/api/ml-lab/modules")
    assert response.status_code == 200
    data = response.json()
    assert "Machine-learning training has not started" in data["disclaimer"]
    for mod in data["planned_modules"]:
        assert mod["status"] == "Planned"

def test_invalid_challan_type():
    response = client.get("/api/mumbai/challans?type=invalid_type")
    assert response.status_code == 400
