"use client";

import { useEffect } from "react";
import Link from "next/link";
import L, { DivIcon } from "leaflet";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polygon,
  Circle,
  ScaleControl,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { OpenInFull } from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import {
  forecast,
  LEVELS,
  SiteView,
  formatLongDayTime,
} from "../../lib/forecast";

interface MapLeafletProps {
  sites: SiteView[];
  threshold: number;
  showSites: boolean;
  showFRAP: boolean;
  showBoundingBox: boolean;
  onSelectSite?: (id: string) => void;
  /** When set, shows a control linking to the full-width map page. */
  expandHref?: string;
}

// Custom marker icon (cached per colour)
const iconCache = new Map<string, DivIcon>();
const createCustomIcon = (color: string) => {
  let icon = iconCache.get(color);
  if (!icon) {
    icon = new DivIcon({
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
      iconSize: [26, 26],
      iconAnchor: [13, 13],
      popupAnchor: [0, -13],
    });
    iconCache.set(color, icon);
  }
  return icon;
};

// Map center and zoom for the Tri-County area
const MAP_CENTER: [number, number] = [34.35, -118.75];
const MAP_ZOOM = 8;

// Tri-County bounding box [minLon, minLat, maxLon, maxLat]
const TRI_COUNTY_BBOX = forecast.region.bbox;
const createBoundingBoxPolygon = (): [number, number][] => {
  const [minLon, minLat, maxLon, maxLat] = TRI_COUNTY_BBOX;
  return [
    [minLat, minLon],
    [minLat, maxLon],
    [maxLat, maxLon],
    [maxLat, minLon],
  ];
};

function RiskLegend({ fontFamily }: { fontFamily?: string }) {
  const map = useMap();

  useEffect(() => {
    const legend = new L.Control({ position: "topright" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "risk-legend");
      div.style.cssText = [
        "background: rgba(255,255,255,0.95)",
        "padding: 8px 10px",
        "border-radius: 4px",
        "border: 1px solid rgba(0,0,0,0.15)",
        "min-width: 160px",
        "font-size: 12px",
        "line-height: 18px",
        "color: #212121",
        `font-family: ${fontFamily ?? "inherit"}`,
      ].join(";");
      const rows = LEVELS.map(
        (l) =>
          `<div style="display:flex;align-items:center;gap:6px">` +
          `<span style="width:10px;height:10px;border-radius:50%;background:${l.color};box-shadow:0 0 0 1.5px #fff"></span>` +
          `<span style="flex:1">${l.level}</span>` +
          `<span style="color:#666;margin-left:12px">${l.range.replace("<", "&lt;")}</span></div>`
      ).join("");
      div.innerHTML = `<div style="font-weight:700;margin-bottom:2px">Fire risk (peak)</div>${rows}`;
      return div;
    };
    legend.addTo(map);
    return () => {
      legend.remove();
    };
  }, [map, fontFamily]);

  return null;
}

function SitePopup({ site, fontFamily }: { site: SiteView; fontFamily?: string }) {
  const w = site.peakWeather;
  const muted = { color: "#666", fontSize: 11 };
  return (
    <div style={{ fontFamily, minWidth: 150, color: "#212121" }}>
      <div style={{ fontWeight: 700, fontSize: 13 }}>{site.name}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "6px 0 8px" }}>
        <span
          style={{
            background: site.color,
            color: site.level === "Moderate" ? "#212121" : "#fff",
            borderRadius: 9,
            padding: "1px 8px",
            fontSize: 10.5,
            fontWeight: 600,
          }}
        >
          {site.level}
        </span>
        <span style={{ fontSize: 11.5, fontWeight: 600 }}>
          Peak risk {site.peak.toFixed(2)}
        </span>
      </div>
      <div style={muted}>{formatLongDayTime(site.peakTime)}</div>
      <div style={muted}>
        {w.temperature_2m.toFixed(1)} °C · RH {w.relative_humidity_2m.toFixed(1)} %
      </div>
    </div>
  );
}

export default function MapLeaflet({
  sites,
  threshold,
  showSites,
  showFRAP,
  showBoundingBox,
  onSelectSite,
  expandHref,
}: MapLeafletProps) {
  const theme = useTheme();
  const fontFamily = theme.typography.fontFamily;

  return (
    <div style={{ position: "relative", height: "100%", width: "100%" }}>
      <MapContainer
        center={MAP_CENTER}
        zoom={MAP_ZOOM}
        scrollWheelZoom={false}
        style={{ height: "100%", width: "100%" }}
      >
        {/* Base tile layer */}
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <ScaleControl position="bottomleft" imperial={false} />
        <RiskLegend fontFamily={fontFamily} />

        {/* Risk overlay around sites above the decision threshold */}
        {showSites &&
          sites
            .filter((s) => s.peak >= threshold)
            .map((s) => (
              <Circle
                key={`risk-${s.id}`}
                center={[s.lat, s.lon]}
                radius={(10 + 25 * s.peak) * 1000}
                interactive={false}
                pathOptions={{
                  color: s.color,
                  weight: 1.5,
                  opacity: 0.55,
                  fillColor: s.color,
                  fillOpacity: 0.16,
                }}
              />
            ))}

        {/* Area of Interest Bounding Box */}
        {showBoundingBox && (
          <Polygon
            positions={createBoundingBoxPolygon()}
            pathOptions={{
              color: "#1976d2",
              weight: 3,
              opacity: 0.7,
              fillColor: "#1976d2",
              fillOpacity: 0.08,
            }}
          >
            <Popup>
              <div style={{ fontFamily, minWidth: 200 }}>
                <div style={{ fontWeight: 700, color: "#1976d2", marginBottom: 6 }}>
                  Area of Interest
                </div>
                <div>
                  <strong>Region:</strong> {forecast.region.name}
                </div>
                <div>
                  <strong>Counties:</strong> {forecast.region.counties.join(", ")}
                </div>
                <div style={{ fontSize: 12, fontFamily: "monospace", marginTop: 4 }}>
                  SW: {TRI_COUNTY_BBOX[1].toFixed(3)}°, {TRI_COUNTY_BBOX[0].toFixed(3)}°
                  <br />
                  NE: {TRI_COUNTY_BBOX[3].toFixed(3)}°, {TRI_COUNTY_BBOX[2].toFixed(3)}°
                </div>
              </div>
            </Popup>
          </Polygon>
        )}

        {/* Sites layer */}
        {showSites &&
          sites.map((site) => (
            <Marker
              key={site.id}
              position={[site.lat, site.lon]}
              icon={createCustomIcon(site.color)}
              title={site.name}
              eventHandlers={{ click: () => onSelectSite?.(site.id) }}
            >
              <Popup>
                <SitePopup site={site} fontFamily={fontFamily} />
              </Popup>
            </Marker>
          ))}
      </MapContainer>

      {showFRAP && (
        <div
          role="status"
          style={{
            position: "absolute",
            top: 10,
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "rgba(255, 152, 0, 0.92)",
            color: "white",
            padding: "6px 12px",
            borderRadius: 4,
            fontSize: 13,
            zIndex: 1000,
            fontFamily,
          }}
        >
          FRAP fire perimeters are not included in the sample data
        </div>
      )}

      {expandHref && (
        <Link
          href={expandHref}
          aria-label="Open the full-width map"
          title="Open the full-width map"
          style={{
            position: "absolute",
            top: 82,
            left: 10,
            zIndex: 1000,
            width: 34,
            height: 34,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#fff",
            backgroundClip: "padding-box",
            border: "2px solid rgba(0,0,0,0.2)",
            borderRadius: 4,
            color: "#000",
          }}
        >
          <OpenInFull sx={{ fontSize: 16 }} />
        </Link>
      )}
    </div>
  );
}
