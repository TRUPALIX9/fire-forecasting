import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppBar, Toolbar, Typography, Container, Box } from "@mui/material";
import Providers from "./providers";
import Link from "next/link";

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
          <AppBar position="static" color="primary">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                🔥 Fire Forecasting
              </Typography>
              <Box sx={{ display: "flex", gap: 2 }}>
                <Link
                  href="/"
                  style={{ color: "white", textDecoration: "none" }}
                >
                  Dashboard
                </Link>
                <Link
                  href="/map"
                  style={{ color: "white", textDecoration: "none" }}
                >
                  Map
                </Link>
                <Link
                  href="/sites"
                  style={{ color: "white", textDecoration: "none" }}
                >
                  Sites
                </Link>
                <Link
                  href="/about"
                  style={{ color: "white", textDecoration: "none" }}
                >
                  About
                </Link>
              </Box>
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
