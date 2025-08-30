import type { Metadata } from "next";
import { Inter } from "next/font/google";
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
  Chip,
} from "@mui/material";
import Providers from "./providers";
import Link from "next/link";
import {
  Map,
  LocationOn,
  Checklist,
  Info,
  LocalFireDepartment,
} from "@mui/icons-material";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Fire Forecasting Dashboard",
  description: "ML-powered wildfire prediction system for Tri-County area",
};

// Theme moved to client Providers

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <AppBar
            position="static"
            elevation={4}
            sx={{
              background: "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)",
            }}
          >
            <Toolbar sx={{ minHeight: 70 }}>
              {/* Logo/Home Button */}
              <Link
                href="/"
                style={{ textDecoration: "none", color: "inherit" }}
              >
                <Button
                  startIcon={
                    <LocalFireDepartment
                      sx={{ fontSize: 28, color: "#ff9800" }}
                    />
                  }
                  sx={{
                    color: "white",
                    fontSize: "1.5rem",
                    fontWeight: 700,
                    textTransform: "none",
                    mr: 4,
                    "&:hover": {
                      background: "rgba(255, 255, 255, 0.1)",
                      transform: "scale(1.02)",
                    },
                    transition: "all 0.2s ease-in-out",
                  }}
                >
                  Fire Forecasting
                </Button>
              </Link>

              {/* Navigation Links */}
              <Box
                sx={{
                  display: "flex",
                  gap: 1,
                  ml: "auto", // Push to the right
                  justifyContent: "flex-end",
                }}
              >
                <Link href="/map" style={{ textDecoration: "none" }}>
                  <Button
                    startIcon={<Map />}
                    sx={{
                      color: "white",
                      textTransform: "none",
                      fontWeight: 500,
                      "&:hover": {
                        background: "rgba(255, 255, 255, 0.1)",
                      },
                    }}
                  >
                    Map
                  </Button>
                </Link>
                <Link href="/sites" style={{ textDecoration: "none" }}>
                  <Button
                    startIcon={<LocationOn />}
                    sx={{
                      color: "white",
                      textTransform: "none",
                      fontWeight: 500,
                      "&:hover": {
                        background: "rgba(255, 255, 255, 0.1)",
                      },
                    }}
                  >
                    Sites
                  </Button>
                </Link>
                <Link href="/checklist" style={{ textDecoration: "none" }}>
                  <Button
                    startIcon={<Checklist />}
                    sx={{
                      color: "white",
                      textTransform: "none",
                      fontWeight: 500,
                      "&:hover": {
                        background: "rgba(255, 255, 255, 0.1)",
                      },
                    }}
                  >
                    Checklist
                  </Button>
                </Link>
                <Link href="/about" style={{ textDecoration: "none" }}>
                  <Button
                    startIcon={<Info />}
                    sx={{
                      color: "white",
                      textTransform: "none",
                      fontWeight: 500,
                      "&:hover": {
                        background: "rgba(255, 255, 255, 0.1)",
                      },
                    }}
                  >
                    About
                  </Button>
                </Link>
                <Link
                  href="/fire-forecasting"
                  style={{ textDecoration: "none" }}
                >
                  <Button
                    startIcon={<LocalFireDepartment />}
                    sx={{
                      color: "white",
                      textTransform: "none",
                      fontWeight: 500,
                      "&:hover": {
                        background: "rgba(255, 255, 255, 0.1)",
                      },
                    }}
                  >
                    Fire Forecast
                  </Button>
                </Link>
              </Box>

              {/* Status Indicator - Removed */}
            </Toolbar>
          </AppBar>
          <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
            {children}
          </Container>
        </Providers>
      </body>
    </html>
  );
}
