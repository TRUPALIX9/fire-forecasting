import { useState, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export interface FRAPFire {
  name: string;
  year: number;
  size: number;
  county: string;
  status: string;
  details: string;
  agency: string;
  cause: string;
}

// Hook to fetch and filter FRAP data
export const useFRAPData = (
  startDate: Date | null,
  endDate: Date | null,
  dateFilterEnabled: boolean,
  fireSizeFilter: [number, number]
) => {
  const [frapData, setFrapData] = useState<FRAPFire[]>([]);
  const [filteredData, setFilteredData] = useState<FRAPFire[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch FRAP data
  useEffect(() => {
    const fetchFRAPData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE}/api/geo/frap`);
        if (response.ok) {
          const data = await response.json();

          // Transform GeoJSON features to table format
          const fires: FRAPFire[] =
            data.features?.map((feature: any) => {
              const props = feature.properties;
              const fireName = props.FIRE_NAME || "Unnamed Fire";
              const year = props.YEAR_ || new Date().getFullYear();
              const size = props.GIS_ACRES || 0;
              const agency = props.AGENCY || "Unknown";

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

              const cause = props.CAUSE
                ? causeMap[props.CAUSE] || `Code ${props.CAUSE}`
                : "Unknown";

              return {
                name: fireName,
                year: year,
                size: size,
                county: "Tri-County Area", // All fires are in Tri-County area
                status: "Contained", // Historical fires are all contained
                details: `${agency} • ${cause}`,
                agency: agency,
                cause: cause,
              };
            }) || [];

          setFrapData(fires);
          setError(null);
        } else {
          setError("Failed to fetch FRAP data");
        }
      } catch (err) {
        setError("Error fetching FRAP data");
        console.error("FRAP fetch error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchFRAPData();
  }, []);

  // Filter data based on current filters
  useEffect(() => {
    if (frapData.length === 0) {
      setFilteredData([]);
      return;
    }

    let filtered = frapData;

    // Date filtering
    if (dateFilterEnabled && startDate && endDate) {
      filtered = filtered.filter((fire: FRAPFire) => {
        const fireDate = new Date(fire.year, 0, 1); // January 1st of fire year
        return fireDate >= startDate && fireDate <= endDate;
      });
    }

    // Fire size filtering
    filtered = filtered.filter(
      (fire: FRAPFire) =>
        fire.size >= fireSizeFilter[0] && fire.size <= fireSizeFilter[1]
    );

    // Sort by size (largest first)
    filtered.sort((a: FRAPFire, b: FRAPFire) => b.size - a.size);

    setFilteredData(filtered);

    console.log(
      `🔥 Fire list filtering: ${frapData.length} total fires, ${filtered.length} after filtering`
    );
  }, [frapData, startDate, endDate, dateFilterEnabled, fireSizeFilter]);

  return { filteredData, loading, error };
};
