"use client";

import { useEffect, useState, useRef } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  GeoJSON,
  Polygon,
  useMap,
} from "react-leaflet";
import { Icon, DivIcon } from "leaflet";
import "leaflet/dist/leaflet.css";

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface Site {
  site: string;
  lat: number;
  lon: number;
}

interface FRAPFeature {
  type: string;
  geometry: any;
  properties: {
    FIRE_NAME?: string;
    YEAR_?: number;
    GIS_ACRES?: number;
    UNIT_ID?: string;
    AGENCY?: string;
    CAUSE?: number;
    [key: string]: any;
  };
}

interface MapLeafletProps {
  showSites: boolean;
  showFRAP: boolean;
  showBoundingBox: boolean;
  startDate?: Date | null;
  endDate?: Date | null;
  dateFilterEnabled?: boolean;
  confidenceThreshold?: number;
  fireSizeFilter?: [number, number];
  mapOpacity?: number;
  frapOpacity?: number;
  sitesOpacity?: number;
  boundingBoxOpacity?: number;
  mapView?: string;
  autoFitBounds?: boolean;
}

// Custom marker icon
const createCustomIcon = (color: string) => {
  return new DivIcon({
    html: `
      <div style="
        background-color: ${color};
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
      "></div>
    `,
    className: "custom-marker",
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
};

// Map center and zoom for Tri-County area
const MAP_CENTER = [34.2, -119.0];
const MAP_ZOOM = 8;

// Tri-County bounding box coordinates [minLon, minLat, maxLon, maxLat]
const TRI_COUNTY_BBOX = [-119.828, 33.422, -117.274, 34.931];

// Create bounding box polygon for visualization
const createBoundingBoxPolygon = (): [number, number][] => {
  const [minLon, minLat, maxLon, maxLat] = TRI_COUNTY_BBOX;
  return [
    [minLat, minLon], // bottom-left
    [minLat, maxLon], // bottom-right
    [maxLat, maxLon], // top-right
    [maxLat, minLon], // top-left
    [minLat, minLon], // close polygon
  ];
};

// Filter FRAP features based on current filters
const filterFRAPFeatures = (
  features: FRAPFeature[],
  startDate: Date | null,
  endDate: Date | null,
  dateFilterEnabled: boolean,
  fireSizeFilter: [number, number]
): FRAPFeature[] => {
  if (!features || features.length === 0) return [];

  return features.filter((feature) => {
    const properties = feature.properties;
    if (!properties) return false;

    // Date filtering
    if (dateFilterEnabled && startDate && endDate) {
      const fireYear = properties.YEAR_;
      if (fireYear) {
        const fireDate = new Date(fireYear, 0, 1); // January 1st of fire year
        if (fireDate < startDate || fireDate > endDate) {
          return false;
        }
      }
    }

    // Fire size filtering
    const fireSize = properties.GIS_ACRES;
    if (fireSize !== undefined && fireSize !== null) {
      if (fireSize < fireSizeFilter[0] || fireSize > fireSizeFilter[1]) {
        return false;
      }
    }

    return true;
  });
};

function MapUpdater({
  showSites,
  showFRAP,
  showBoundingBox,
  startDate,
  endDate,
  dateFilterEnabled,
  confidenceThreshold,
  fireSizeFilter,
  mapOpacity,
  frapOpacity,
  sitesOpacity,
  boundingBoxOpacity,
  mapView,
  autoFitBounds,
}: MapLeafletProps) {
  const map = useMap();

  useEffect(() => {
    // Fit bounds when layers change
    if (showSites || showFRAP || showBoundingBox) {
      map.invalidateSize();
    }

    // Handle auto-fit bounds
    if (autoFitBounds && (showSites || showFRAP)) {
      // This would be implemented to automatically fit the map to show all visible data
      map.invalidateSize();
    }
  }, [showSites, showFRAP, showBoundingBox, autoFitBounds, map]);

  return null;
}

export default function MapLeaflet({
  showSites,
  showFRAP,
  showBoundingBox,
  startDate,
  endDate,
  dateFilterEnabled = false,
  confidenceThreshold = 20,
  fireSizeFilter = [0, 100000],
  mapOpacity = 0.8,
  frapOpacity = 0.6,
  sitesOpacity = 0.9,
  boundingBoxOpacity = 0.7,
  mapView = "standard",
  autoFitBounds = true,
}: MapLeafletProps) {
  const [sites, setSites] = useState<Site[]>([]);
  const [frapData, setFrapData] = useState<any>(null);
  const [filteredFrapData, setFilteredFrapData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Filter FRAP data when filters change
  useEffect(() => {
    if (frapData && frapData.features) {
      const filteredFeatures = filterFRAPFeatures(
        frapData.features,
        startDate || null,
        endDate || null,
        dateFilterEnabled,
        fireSizeFilter
      );

      setFilteredFrapData({
        ...frapData,
        features: filteredFeatures,
      });

      // Show a more detailed breakdown
      if (dateFilterEnabled && startDate && endDate) {
        const yearFiltered = frapData.features.filter((f: FRAPFeature) => {
          const year = f.properties?.YEAR_;
          if (year) {
            const fireDate = new Date(year, 0, 1);
            return fireDate >= startDate && fireDate <= endDate;
          }
          return false;
        });
        console.log(
          `📅 Date filtering: ${yearFiltered.length} features within date range`
        );
      }

      if (fireSizeFilter[0] > 0 || fireSizeFilter[1] < 100000) {
        const sizeFiltered = frapData.features.filter((f: FRAPFeature) => {
          const size = f.properties?.GIS_ACRES;
          return (
            size !== undefined &&
            size >= fireSizeFilter[0] &&
            size <= fireSizeFilter[1]
          );
        });
        console.log(
          `🔥 Size filtering: ${sizeFiltered.length} features within size range`
        );
      }
    }
  }, [frapData, startDate, endDate, dateFilterEnabled, fireSizeFilter]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch sites
        if (showSites) {
          const sitesResponse = await fetch(`${API_BASE}/api/geo/sites`);
          if (sitesResponse.ok) {
            const sitesData = await sitesResponse.json();
            // Extract site properties from GeoJSON features
            const siteFeatures = sitesData.features || [];
            const extractedSites = siteFeatures.map((feature: any) => ({
              site: feature.properties?.site || "Unknown Site",
              lat:
                feature.properties?.lat ||
                feature.geometry?.coordinates?.[1] ||
                0,
              lon:
                feature.properties?.lon ||
                feature.geometry?.coordinates?.[0] ||
                0,
            }));
            setSites(extractedSites);
          }
        }

        // Fetch FRAP data
        if (showFRAP) {
          try {
            const frapResponse = await fetch(`${API_BASE}/api/geo/frap`);
            if (frapResponse.ok) {
              const frapData = await frapResponse.json();
              setFrapData(frapData);
            } else {
              console.warn("FRAP data not available:", frapResponse.status);
            }
          } catch (error) {
            console.warn("Failed to fetch FRAP data:", error);
          }
        }
      } catch (error) {
        console.error("Error fetching map data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [showSites, showFRAP]);

  if (loading) {
    return (
      <div
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#f5f5f5",
        }}
      >
        Loading map...
      </div>
    );
  }

  return (
    <MapContainer
      center={MAP_CENTER as [number, number]}
      zoom={MAP_ZOOM}
      style={{ height: "100%", width: "100%" }}
    >
      <MapUpdater
        showSites={showSites}
        showFRAP={showFRAP}
        showBoundingBox={showBoundingBox}
        startDate={startDate}
        endDate={endDate}
        dateFilterEnabled={dateFilterEnabled}
        confidenceThreshold={confidenceThreshold}
        fireSizeFilter={fireSizeFilter}
        mapOpacity={mapOpacity}
        frapOpacity={frapOpacity}
        sitesOpacity={sitesOpacity}
        boundingBoxOpacity={boundingBoxOpacity}
        mapView={mapView}
        autoFitBounds={autoFitBounds}
      />

      {/* Base tile layer */}
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />

      {/* Area of Interest Bounding Box */}
      {showBoundingBox && (
        <Polygon
          positions={createBoundingBoxPolygon()}
          pathOptions={{
            color: "#1976d2",
            weight: 3,
            opacity: boundingBoxOpacity,
            fillColor: "#1976d2",
            fillOpacity: boundingBoxOpacity * 0.15,
          }}
        >
          <Popup>
            <div style={{ minWidth: "200px" }}>
              <h3 style={{ margin: "0 0 8px 0", color: "#1976d2" }}>
                Area of Interest
              </h3>
              <p style={{ margin: "4px 0" }}>
                <strong>Region:</strong> Tri-County Area
              </p>
              <p style={{ margin: "4px 0" }}>
                <strong>Counties:</strong> Ventura, Santa Barbara, Los Angeles
              </p>
              <p style={{ margin: "4px 0" }}>
                <strong>Coordinates:</strong>
              </p>
              <p
                style={{
                  margin: "4px 0",
                  fontSize: "12px",
                  fontFamily: "monospace",
                }}
              >
                SW: {TRI_COUNTY_BBOX[1].toFixed(3)}°,{" "}
                {TRI_COUNTY_BBOX[0].toFixed(3)}°
              </p>
              <p
                style={{
                  margin: "4px 0",
                  fontSize: "12px",
                  fontFamily: "monospace",
                }}
              >
                NE: {TRI_COUNTY_BBOX[3].toFixed(3)}°,{" "}
                {TRI_COUNTY_BBOX[2].toFixed(3)}°
              </p>
              <p style={{ margin: "4px 0", fontSize: "12px", color: "#666" }}>
                This area contains 20+ WUI monitoring sites and historical fire
                data
              </p>
            </div>
          </Popup>
        </Polygon>
      )}

      {/* Sites layer */}
      {showSites &&
        sites.map((site, index) => {
          // Validate site data before rendering
          if (
            !site ||
            typeof site.lat !== "number" ||
            typeof site.lon !== "number"
          ) {
            console.warn("Invalid site data:", site);
            return null;
          }

          return (
            <Marker
              key={index}
              position={[site.lat, site.lon]}
              icon={createCustomIcon("#d32f2f")}
              opacity={sitesOpacity}
            >
              <Popup>
                <div>
                  <h3>{site.site || "Unknown Site"}</h3>
                  <p>Lat: {site.lat ? site.lat.toFixed(4) : "Unknown"}</p>
                  <p>Lon: {site.lon ? site.lon.toFixed(4) : "Unknown"}</p>
                </div>
              </Popup>
            </Marker>
          );
        })}

      {/* FRAP layer */}
      {showFRAP && (
        <>
          {!frapData && (
            <div
              style={{
                position: "absolute",
                top: "10px",
                right: "10px",
                backgroundColor: "rgba(255, 152, 0, 0.9)",
                color: "white",
                padding: "8px 12px",
                borderRadius: "4px",
                fontSize: "14px",
                zIndex: 1000,
              }}
            >
              Loading fire perimeters...
            </div>
          )}
          {filteredFrapData && (
            <>
              {/* Filter status indicator */}
              {(dateFilterEnabled ||
                fireSizeFilter[0] > 0 ||
                fireSizeFilter[1] < 100000) && (
                <div
                  style={{
                    position: "absolute",
                    top: "10px",
                    left: "10px",
                    backgroundColor: "rgba(25, 118, 210, 0.9)",
                    color: "white",
                    padding: "8px 12px",
                    borderRadius: "4px",
                    fontSize: "12px",
                    zIndex: 1000,
                    maxWidth: "300px",
                  }}
                >
                  <div style={{ fontWeight: "bold", marginBottom: "2px" }}>
                    🔍 Filters Active
                  </div>
                  <div>
                    Showing {filteredFrapData.features.length} of{" "}
                    {frapData?.features?.length || 0} fires
                  </div>
                  {dateFilterEnabled && startDate && endDate && (
                    <div style={{ fontSize: "11px", opacity: 0.9 }}>
                      Date: {startDate.toISOString().split("T")[0]} to{" "}
                      {endDate.toISOString().split("T")[0]}
                    </div>
                  )}
                  {(fireSizeFilter[0] > 0 || fireSizeFilter[1] < 100000) && (
                    <div style={{ fontSize: "11px", opacity: 0.9 }}>
                      Size: {fireSizeFilter[0].toLocaleString()} -{" "}
                      {fireSizeFilter[1].toLocaleString()} acres
                    </div>
                  )}
                </div>
              )}
              <GeoJSON
                data={filteredFrapData}
                style={(feature) => ({
                  color: "#ff9800",
                  weight: 2,
                  opacity: frapOpacity,
                  fillColor: "#ff9800",
                  fillOpacity: frapOpacity * 0.3,
                })}
                onEachFeature={(feature, layer) => {
                  if (feature.properties) {
                    const {
                      FIRE_NAME,
                      YEAR_,
                      GIS_ACRES,
                      UNIT_ID,
                      AGENCY,
                      CAUSE,
                    } = feature.properties;

                    // Format fire name (handle null/undefined)
                    const fireName = FIRE_NAME || "Unnamed Fire";
                    const year = YEAR_ || "Unknown";
                    const acres = GIS_ACRES
                      ? `${GIS_ACRES.toLocaleString()} acres`
                      : "Unknown size";
                    const unit = UNIT_ID || "Unknown";
                    const agency = AGENCY || "Unknown";

                    // Map cause codes to readable text
                    const causeMap: { [key: number]: string } = {
                      1: "Lightning",
                      2: "Equipment Use",
                      3: "Smoking",
                      4: "Campfire",
                      5: "Debris Burning",
                      6: "Railroad",
                      7: "Arson",
                      8: "Playing with Fire",
                      9: "Miscellaneous",
                      10: "Vehicle",
                      11: "Powerline",
                      12: "Firefighter Training",
                      13: "Non-Fire",
                      14: "Unknown",
                      15: "Structure",
                      16: "Fireworks",
                      17: "Escaped Prescribed Burn",
                      18: "Illegal Alien Campfire",
                      19: "Firearms",
                      20: "Explosives",
                      21: "Welding",
                      22: "Cutting/Grinding",
                      23: "Gasoline",
                      24: "Spontaneous Combustion",
                      25: "Children",
                      26: "Firearms",
                      27: "Fireworks",
                      28: "Firearms",
                      29: "Fireworks",
                      30: "Firearms",
                    };

                    const cause = CAUSE
                      ? causeMap[CAUSE] || `Code ${CAUSE}`
                      : "Unknown";

                    layer.bindPopup(`
                    <div style="min-width: 200px;">
                      <h3 style="margin: 0 0 8px 0; color: #d32f2f;">${fireName}</h3>
                      <p style="margin: 4px 0;"><strong>Year:</strong> ${year}</p>
                      <p style="margin: 4px 0;"><strong>Size:</strong> ${acres}</p>
                      <p style="margin: 4px 0;"><strong>Unit:</strong> ${unit}</p>
                      <p style="margin: 4px 0;"><strong>Agency:</strong> ${agency}</p>
                      <p style="margin: 4px 0;"><strong>Cause:</strong> ${cause}</p>
                    </div>
                  `);
                  }
                }}
              />
            </>
          )}
        </>
      )}
    </MapContainer>
  );
}
