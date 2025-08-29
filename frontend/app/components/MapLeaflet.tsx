'use client'

import { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, GeoJSON, useMap } from 'react-leaflet'
import { Icon, DivIcon } from 'leaflet'
import 'leaflet/dist/leaflet.css'

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

interface Site {
  site: string
  lat: number
  lon: number
}

interface FRAPFeature {
  type: string
  geometry: any
  properties: {
    fire_name: string
    year: number
    acres: number
    county: string
  }
}

interface MapLeafletProps {
  showSites: boolean
  showFRAP: boolean
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
    className: 'custom-marker',
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  })
}

// Map center and zoom for Tri-County area
const MAP_CENTER = [34.2, -119.0]
const MAP_ZOOM = 8

function MapUpdater({ showSites, showFRAP }: MapLeafletProps) {
  const map = useMap()
  
  useEffect(() => {
    // Fit bounds when layers change
    if (showSites || showFRAP) {
      map.invalidateSize()
    }
  }, [showSites, showFRAP, map])
  
  return null
}

export default function MapLeaflet({ showSites, showFRAP }: MapLeafletProps) {
  const [sites, setSites] = useState<Site[]>([])
  const [frapData, setFrapData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch sites
        if (showSites) {
          const sitesResponse = await fetch(`${API_BASE}/api/geo/sites`)
          if (sitesResponse.ok) {
            const sitesData = await sitesResponse.json()
            setSites(sitesData.features || [])
          }
        }
        
        // Fetch FRAP data
        if (showFRAP) {
          const frapResponse = await fetch(`${API_BASE}/api/geo/frap`)
          if (frapResponse.ok) {
            const frapData = await frapResponse.json()
            setFrapData(frapData)
          }
        }
      } catch (error) {
        console.error('Error fetching map data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [showSites, showFRAP])

  if (loading) {
    return (
      <div style={{ 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#f5f5f5'
      }}>
        Loading map...
      </div>
    )
  }

  return (
    <MapContainer
      center={MAP_CENTER as [number, number]}
      zoom={MAP_ZOOM}
      style={{ height: '100%', width: '100%' }}
    >
      <MapUpdater showSites={showSites} showFRAP={showFRAP} />
      
      {/* Base tile layer */}
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      
      {/* Sites layer */}
      {showSites && sites.map((site, index) => (
        <Marker
          key={index}
          position={[site.lat, site.lon]}
          icon={createCustomIcon('#d32f2f')}
        >
          <Popup>
            <div>
              <h3>{site.site}</h3>
              <p>Lat: {site.lat.toFixed(4)}</p>
              <p>Lon: {site.lon.toFixed(4)}</p>
            </div>
          </Popup>
        </Marker>
      ))}
      
      {/* FRAP layer */}
      {showFRAP && frapData && (
        <GeoJSON
          data={frapData}
          style={(feature) => ({
            color: '#ff9800',
            weight: 2,
            opacity: 0.8,
            fillColor: '#ff9800',
            fillOpacity: 0.2
          })}
          onEachFeature={(feature, layer) => {
            if (feature.properties) {
              const { fire_name, year, acres, county } = feature.properties
              layer.bindPopup(`
                <div>
                  <h3>${fire_name}</h3>
                  <p><strong>Year:</strong> ${year}</p>
                  <p><strong>Acres:</strong> ${acres.toLocaleString()}</p>
                  <p><strong>County:</strong> ${county}</p>
                </div>
              `)
            }
          }}
        />
      )}
    </MapContainer>
  )
}
