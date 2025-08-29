/**
 * API client for communicating with the FastAPI backend.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export interface StatusResponse {
  status: string;
  timestamp: string;
  last_run?: string;
  rows_total?: number;
  model?: string;
}

export interface GlobalMetrics {
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1_score: number;
  threshold: number;
  train_time_s: number;
  infer_time_s: number;
}

export interface SiteMetrics {
  site: string;
  pr_auc: number;
  f1_at_tau: number;
  positives: number;
  n: number;
}

export interface SitesMetricsResponse {
  sites: SiteMetrics[];
  summary: Record<string, any>;
}

export interface CurveData {
  x_values: number[];
  y_values: number[];
  thresholds: number[];
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

export interface GeoPoint {
  type: string;
  geometry: {
    type: string;
    coordinates: [number, number];
  };
  properties: Record<string, any>;
}

export interface GeoCollection {
  type: string;
  features: GeoPoint[];
}

export interface TrainingRequest {
  config_override?: Record<string, any>;
}

export interface TrainingResponse {
  status: string;
  message: string;
  metrics?: GlobalMetrics;
  error?: string;
}

/**
 * Generic API request function with error handling.
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API request failed for ${endpoint}:`, error);
    throw error;
  }
}

/**
 * API client functions.
 */
export const apiClient = {
  // Status and configuration
  async getStatus(): Promise<StatusResponse> {
    return apiRequest<StatusResponse>("/api/status");
  },

  async getConfig(): Promise<Record<string, any>> {
    return apiRequest<Record<string, any>>("/api/config");
  },

  // Training
  async trainModel(request: TrainingRequest = {}): Promise<TrainingResponse> {
    return apiRequest<TrainingResponse>("/api/train", {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  // Metrics
  async getGlobalMetrics(): Promise<GlobalMetrics> {
    return apiRequest<GlobalMetrics>("/api/metrics/global");
  },

  async getSitesMetrics(): Promise<SitesMetricsResponse> {
    return apiRequest<SitesMetricsResponse>("/api/metrics/sites");
  },

  // Curves
  async getPRCurve(): Promise<CurveData> {
    return apiRequest<CurveData>("/api/curves/pr");
  },

  async getROCCurve(): Promise<CurveData> {
    return apiRequest<CurveData>("/api/curves/roc");
  },

  // Confusion matrix
  async getConfusionMatrix(threshold: number = 0.5): Promise<ConfusionMatrix> {
    return apiRequest<ConfusionMatrix>(`/api/confusion?threshold=${threshold}`);
  },

  // Geospatial data
  async getSitesGeo(): Promise<GeoCollection> {
    return apiRequest<GeoCollection>("/api/geo/sites");
  },

  async getFRAPGeo(): Promise<GeoCollection> {
    return apiRequest<GeoCollection>("/api/geo/frap");
  },

  // Utility functions
  getArtifactUrl(path: string): string {
    return `${API_BASE_URL}/artifacts/${path}`;
  },

  getDownloadUrl(filename: string): string {
    return `${API_BASE_URL}/artifacts/${filename}`;
  },
};

/**
 * SWR fetcher functions for use with useSWR hook.
 */
export const swrFetchers = {
  status: () => apiClient.getStatus(),
  config: () => apiClient.getConfig(),
  globalMetrics: () => apiClient.getGlobalMetrics(),
  sitesMetrics: () => apiClient.getSitesMetrics(),
  prCurve: () => apiClient.getPRCurve(),
  rocCurve: () => apiClient.getROCCurve(),
  confusionMatrix: (threshold: number) =>
    apiClient.getConfusionMatrix(threshold),
  sitesGeo: () => apiClient.getSitesGeo(),
  frapGeo: () => apiClient.getFRAPGeo(),
};

/**
 * Error handling utilities.
 */
export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public endpoint?: string
  ) {
    super(message);
    this.name = "APIError";
  }
}

/**
 * Check if the backend is available.
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    await apiClient.getStatus();
    return true;
  } catch (error) {
    console.warn("Backend health check failed:", error);
    return false;
  }
}

/**
 * Retry function with exponential backoff.
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === maxRetries) {
        break;
      }

      const delay = baseDelay * Math.pow(2, attempt);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

export default apiClient;
