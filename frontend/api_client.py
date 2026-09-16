import requests


class ApiError(Exception):
    pass


def ask_question(question: str, api_base_url: str, timeout: int = 60):
    """Call the backend /query endpoint and return (answer, sources)."""
    url = f"{api_base_url.rstrip('/')}/query"
    try:
        response = requests.post(url, json={"question": question}, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ApiError(f"Could not reach the API at {url}: {e}") from e

    data = response.json()
    return data["answer"], data["sources"]


def check_health(api_base_url: str, timeout: int = 5) -> bool:
    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
