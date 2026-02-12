import { HeadContent, Outlet, Scripts, createRootRoute } from '@tanstack/react-router'
import { ConvexProvider, ConvexReactClient } from 'convex/react'
import { Toaster } from 'sonner'

import { AppSidebar } from '../components/app-sidebar'
import { ThemeProvider } from '../components/theme-provider'
import { SidebarProvider, SidebarInset } from '../components/ui/sidebar'
import { useSeedSettings } from '../hooks/use-seed-settings'
import { CONVEX_URL } from '../lib/api-urls'

import appCss from '../styles.css?url'

const convex = new ConvexReactClient(CONVEX_URL)

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'CERA - Synthetic ABSA Dataset Generation' },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
      { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
      { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/favicon-16x16.png' },
      { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' },
      { rel: 'apple-touch-icon', sizes: '180x180', href: '/apple-touch-icon.png' },
      { rel: 'manifest', href: '/site.webmanifest' },
    ],
  }),
  component: RootComponent,
})

function RootComponent() {
  return (
    <RootDocument>
      <ThemeProvider
        attribute="class"
        defaultTheme="dark"
        enableSystem
        disableTransitionOnChange
      >
        <ConvexProvider client={convex}>
          <AppShell />
        </ConvexProvider>
      </ThemeProvider>
    </RootDocument>
  )
}

function AppShell() {
  // Seed settings from env vars on first load
  useSeedSettings()

  return (
    <>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
          <Outlet />
        </SidebarInset>
      </SidebarProvider>
      <Toaster position="bottom-right" />
    </>
  )
}

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body className="min-h-screen bg-background font-sans antialiased">
        {children}
        <Scripts />
      </body>
    </html>
  )
}
