"use client";

import { AppBar, Toolbar, Box, Button } from "@mui/material";
import Link from "next/link";
import {
  Home,
  Settings,
  History,
  LocalFireDepartment,
} from "@mui/icons-material";

const navItems = [
  { href: "/", label: "Home", icon: <Home /> },
  { href: "/settings", label: "Settings", icon: <Settings /> },
  { href: "/ml-history", label: "ML History", icon: <History /> },
];

export default function AppHeader() {
  return (
    <AppBar
      position="static"
      elevation={4}
      sx={{
        background: "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)",
      }}
    >
      {/* Responsive value so it overrides the toolbar mixin's sm media query */}
      <Toolbar sx={{ minHeight: { xs: 64, sm: 70 } }}>
        {/* Logo/Home Button */}
        <Button
          component={Link}
          href="/"
          startIcon={
            <LocalFireDepartment sx={{ fontSize: 28, color: "#ff9800" }} />
          }
          sx={{
            color: "white",
            fontSize: { xs: "1.1rem", sm: "1.5rem" },
            fontWeight: 700,
            textTransform: "none",
            whiteSpace: "nowrap",
            mr: { xs: 1, sm: 4 },
            "&:hover": {
              background: "rgba(255, 255, 255, 0.1)",
              transform: "scale(1.02)",
            },
            transition: "all 0.2s ease-in-out",
          }}
        >
          Fire Forecasting
        </Button>

        {/* Navigation Links: icon-only below the sm breakpoint */}
        <Box
          component="nav"
          sx={{
            display: "flex",
            gap: { xs: 0, sm: 1 },
            ml: "auto",
            justifyContent: "flex-end",
          }}
        >
          {navItems.map((item) => (
            <Button
              key={item.href}
              component={Link}
              href={item.href}
              aria-label={item.label}
              startIcon={item.icon}
              sx={{
                color: "white",
                textTransform: "none",
                fontWeight: 500,
                minWidth: { xs: 40, sm: 64 },
                "& .MuiButton-startIcon": {
                  mr: { xs: 0, sm: 1 },
                  ml: { xs: 0, sm: -0.5 },
                },
                "&:hover": {
                  background: "rgba(255, 255, 255, 0.1)",
                },
              }}
            >
              <Box
                component="span"
                sx={{ display: { xs: "none", sm: "inline" } }}
              >
                {item.label}
              </Box>
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
}
