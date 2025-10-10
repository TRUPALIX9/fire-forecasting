import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppBar, Toolbar, Container, Box, Button } from "@mui/material";
import Link from "next/link";
import {
  Home,
  Settings,
  History,
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
        <AppBar
          position="static"
          elevation={4}
          sx={{
            background: "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)",
          }}
        >
          <Toolbar sx={{ minHeight: 70 }}>
            {/* Logo/Home Button */}
            <Link href="/" style={{ textDecoration: "none", color: "inherit" }}>
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
                ml: "auto",
                justifyContent: "flex-end",
              }}
            >
              <Link href="/" style={{ textDecoration: "none" }}>
                <Button
                  startIcon={<Home />}
                  sx={{
                    color: "white",
                    textTransform: "none",
                    fontWeight: 500,
                    "&:hover": {
                      background: "rgba(255, 255, 255, 0.1)",
                    },
                  }}
                >
                  Home
                </Button>
              </Link>
              <Link href="/settings" style={{ textDecoration: "none" }}>
                <Button
                  startIcon={<Settings />}
                  sx={{
                    color: "white",
                    textTransform: "none",
                    fontWeight: 500,
                    "&:hover": {
                      background: "rgba(255, 255, 255, 0.1)",
                    },
                  }}
                >
                  Settings
                </Button>
              </Link>
              <Link href="/ml-history" style={{ textDecoration: "none" }}>
                <Button
                  startIcon={<History />}
                  sx={{
                    color: "white",
                    textTransform: "none",
                    fontWeight: 500,
                    "&:hover": {
                      background: "rgba(255, 255, 255, 0.1)",
                    },
                  }}
                >
                  ML History
                </Button>
              </Link>
            </Box>
          </Toolbar>
        </AppBar>
        <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
          {children}
        </Container>
      </body>
    </html>
  );
}
