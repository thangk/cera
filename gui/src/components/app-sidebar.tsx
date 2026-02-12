import { Link, useRouterState } from '@tanstack/react-router'
import { useState, useEffect } from 'react'
import { useQuery } from 'convex/react'
import { api } from '../../convex/_generated/api'
import {
  LayoutDashboard,
  Sparkles,
  Activity,
  Database,
  Settings,
  ChevronRight,
  Server,
  Eye,
  FlaskConical,
  Info,
} from 'lucide-react'

import { ThemeToggle } from './theme-toggle'
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip'
import { PYTHON_API_URL } from '../lib/api-urls'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from './ui/sidebar'

const navItems = [
  { title: 'Dashboard', href: '/', icon: LayoutDashboard },
  { title: 'Create Job', href: '/create-job', icon: Sparkles },
  { title: 'Jobs', href: '/jobs', icon: Activity },
]

type ServiceStatus = 'online' | 'offline' | 'checking' | 'configured' | 'not_configured'

// Service descriptions for info tooltips
const SERVICE_INFO: Record<string, string> = {
  convex: 'Real-time database for job state, logs, and settings persistence.',
  fastapi: 'Python backend that runs the CERA pipeline (SIL, MAV, generation).',
  openrouter: 'LLM gateway for accessing multiple AI models (Claude, GPT, etc.).',
  tavily: 'Web search API used by SIL for subject research.',
}

function useServiceStatus() {
  const [fastapiStatus, setFastapiStatus] = useState<ServiceStatus>('checking')
  const [openrouterStatus, setOpenrouterStatus] = useState<ServiceStatus>('checking')
  const [tavilyStatus, setTavilyStatus] = useState<ServiceStatus>('checking')

  // Use Convex query for status - if it returns data, Convex is online
  const convexPing = useQuery(api.settings.ping)
  const settings = useQuery(api.settings.get)

  const convexStatus: ServiceStatus = convexPing === undefined ? 'checking' : convexPing?.ok ? 'online' : 'offline'

  useEffect(() => {
    const apiUrl = PYTHON_API_URL

    const checkFastAPI = async () => {
      try {
        const apiRes = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(3000) })
        setFastapiStatus(apiRes.ok ? 'online' : 'offline')
      } catch {
        setFastapiStatus('offline')
      }
    }

    checkFastAPI()
    const interval = setInterval(checkFastAPI, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  // Validate external API keys through FastAPI backend
  useEffect(() => {
    if (settings === undefined) return

    const apiUrl = PYTHON_API_URL

    const validateOpenRouter = async () => {
      if (!settings?.openrouterApiKey) {
        setOpenrouterStatus('not_configured')
        return
      }
      try {
        const res = await fetch(`${apiUrl}/api/validate/openrouter`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ apiKey: settings.openrouterApiKey }),
          signal: AbortSignal.timeout(15000),
        })
        const data = await res.json()
        setOpenrouterStatus(data.valid ? 'online' : 'offline')
      } catch {
        setOpenrouterStatus('offline')
      }
    }

    const validateTavily = async () => {
      if (!settings?.tavilyApiKey) {
        setTavilyStatus('not_configured')
        return
      }
      try {
        const res = await fetch(`${apiUrl}/api/validate/tavily`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ apiKey: settings.tavilyApiKey }),
          signal: AbortSignal.timeout(15000),
        })
        const data = await res.json()
        setTavilyStatus(data.valid ? 'online' : 'offline')
      } catch {
        setTavilyStatus('offline')
      }
    }

    validateOpenRouter()
    validateTavily()

    // Re-validate every 60 seconds
    const interval = setInterval(() => {
      validateOpenRouter()
      validateTavily()
    }, 60000)

    return () => clearInterval(interval)
  }, [settings])

  return { convexStatus, fastapiStatus, openrouterStatus, tavilyStatus }
}

function ServiceRow({
  status,
  label,
  serviceKey
}: {
  status: ServiceStatus
  label: string
  serviceKey: string
}) {
  const colors: Record<ServiceStatus, string> = {
    online: 'bg-green-500',
    offline: 'bg-red-500',
    checking: 'bg-yellow-500 animate-pulse',
    configured: 'bg-green-500',
    not_configured: 'bg-gray-400',
  }

  const statusText: Record<ServiceStatus, string> = {
    online: 'Online',
    offline: 'Offline',
    checking: 'Checking...',
    configured: 'Configured',
    not_configured: 'Not configured',
  }

  return (
    <div className="flex items-center justify-between py-1">
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-1.5 cursor-default">
            <div className={`h-2 w-2 rounded-full ${colors[status]}`} />
            <span className="text-xs text-muted-foreground">{label}</span>
          </div>
        </TooltipTrigger>
        <TooltipContent side="top">
          <p>{label}: {statusText[status]}</p>
        </TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="h-3 w-3 text-muted-foreground/50 hover:text-muted-foreground cursor-help" />
        </TooltipTrigger>
        <TooltipContent side="left" className="max-w-[200px]">
          <p className="text-xs">{SERVICE_INFO[serviceKey]}</p>
        </TooltipContent>
      </Tooltip>
    </div>
  )
}

export function AppSidebar() {
  const router = useRouterState()
  const currentPath = router.location.pathname
  const { convexStatus, fastapiStatus, openrouterStatus, tavilyStatus } = useServiceStatus()

  return (
    <Sidebar>
      <SidebarHeader className="border-b border-sidebar-border">
        <div className="flex items-center gap-3 px-4 py-3">
          <div className="flex h-9 items-center justify-center rounded-lg bg-slate-800 px-2">
            <img
              src="/cera-logo.png"
              alt="CERA"
              className="h-6 w-auto"
            />
          </div>
          <span className="text-xs text-muted-foreground">Synthetic ABSA</span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => {
                const isActive = currentPath === item.href ||
                  (item.href !== '/' && currentPath.startsWith(item.href))

                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild isActive={isActive}>
                      <Link to={item.href} preload="intent">
                        <item.icon className="h-4 w-4" />
                        <span>{item.title}</span>
                        {isActive && (
                          <ChevronRight className="ml-auto h-4 w-4" />
                        )}
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Tools</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={currentPath === '/dataset-tools'}>
                  <Link to="/dataset-tools" search={{ tool: undefined }} preload="intent">
                    <Database className="h-4 w-4" />
                    <span>Dataset Tools</span>
                    {currentPath === '/dataset-tools' && (
                      <ChevronRight className="ml-auto h-4 w-4" />
                    )}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={currentPath === '/viewer-tools'}>
                  <Link to="/viewer-tools" preload="intent">
                    <Eye className="h-4 w-4" />
                    <span>Viewer Tools</span>
                    {currentPath === '/viewer-tools' && (
                      <ChevronRight className="ml-auto h-4 w-4" />
                    )}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={currentPath.startsWith('/research-tools')}>
                  <Link to="/research-tools" preload="intent">
                    <FlaskConical className="h-4 w-4" />
                    <span>Research Tools</span>
                    {currentPath.startsWith('/research-tools') && (
                      <ChevronRight className="ml-auto h-4 w-4" />
                    )}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={currentPath === '/convex'}>
                  <Link to="/convex" preload="intent">
                    <Server className="h-4 w-4" />
                    <span>Convex Dashboard</span>
                    {currentPath === '/convex' && (
                      <ChevronRight className="ml-auto h-4 w-4" />
                    )}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={currentPath === '/settings'}>
                  <Link to="/settings" preload="intent">
                    <Settings className="h-4 w-4" />
                    <span>Settings</span>
                    {currentPath === '/settings' && (
                      <ChevronRight className="ml-auto h-4 w-4" />
                    )}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-sidebar-border">
        <div className="px-4 py-3 space-y-3">
          {/* Services Section */}
          <div className="space-y-2">
            <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Services</span>

            {/* Internal Services */}
            <div className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground/70">Internal</span>
              <ServiceRow status={convexStatus} label="Convex" serviceKey="convex" />
              <ServiceRow status={fastapiStatus} label="FastAPI" serviceKey="fastapi" />
            </div>

            {/* External Services */}
            <div className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground/70">External</span>
              <ServiceRow status={openrouterStatus} label="OpenRouter" serviceKey="openrouter" />
              <ServiceRow status={tavilyStatus} label="Tavily" serviceKey="tavily" />
            </div>
          </div>

          {/* Version & Theme */}
          <div className="flex items-center py-1 pt-2 border-t border-sidebar-border/50">
            <span className="text-xs text-muted-foreground flex-1">CERA v1.0.0</span>
            <div className="mr-[-10px]">
              <ThemeToggle />
            </div>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}
