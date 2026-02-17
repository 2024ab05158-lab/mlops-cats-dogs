import requests

BASE_URL = "http://localhost:8001"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print("✅ Health check passed")

def test_metrics():
    r = requests.get(f"{BASE_URL}/metrics")
    assert r.status_code == 200
    print("✅ Metrics endpoint reachable")

if __name__ == "__main__":
    test_health()
    test_metrics()
    print("🚀 Smoke tests successful")