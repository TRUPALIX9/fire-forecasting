import type { Metadata } from "next";
import { Container } from "@mui/material";
import Providers from "./providers";
import AppHeader from "./components/AppHeader";

export const metadata: Metadata = {
  title: "Fire Forecasting Dashboard",
  description:
    "Wildfire risk dashboard prototype with sample forecasts for the Tri-County area",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <AppHeader />
          <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
            {children}
          </Container>
        </Providers>
      </body>
    </html>
  );
}
