import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { AppBar, Toolbar, Typography, Container, Box } from '@mui/material'
import Link from 'next/link'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Fire Forecasting Dashboard',
  description: 'ML-powered wildfire prediction system for Tri-County area',
}

// Create MUI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#d32f2f', // Fire red
    },
    secondary: {
      main: '#ff9800', // Orange
    },
    background: {
      default: '#fafafa',
    },
  },
  typography: {
    fontFamily: inter.style.fontFamily,
  },
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <AppBar position="static" color="primary">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                🔥 Fire Forecasting
              </Typography>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Link href="/" style={{ color: 'white', textDecoration: 'none' }}>
                  Dashboard
                </Link>
                <Link href="/map" style={{ color: 'white', textDecoration: 'none' }}>
                  Map
                </Link>
                <Link href="/sites" style={{ color: 'white', textDecoration: 'none' }}>
                  Sites
                </Link>
                <Link href="/about" style={{ color: 'white', textDecoration: 'none' }}>
                  About
                </Link>
              </Box>
            </Toolbar>
          </AppBar>
          <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
            {children}
          </Container>
        </ThemeProvider>
      </body>
    </html>
  )
}
