export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    console.error("API request failed", { url: `${API_BASE}${path}`, err });
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

export async function apiPost<T>(path: string, body?: any): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    console.error("API request failed", { url: `${API_BASE}${path}`, err });
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}
