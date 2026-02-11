import { createFileRoute } from '@tanstack/react-router'
import { Copy, Check, ExternalLink, Server, Key, Eye, EyeOff, AlertCircle } from 'lucide-react'
import { useState } from 'react'

import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'

export const Route = createFileRoute('/convex')({
  component: ConvexPage,
})

// Hardcoded localhost URLs for local development
const CONVEX_URL = 'http://localhost:3210'
const CONVEX_DASHBOARD_URL = 'http://localhost:6791'

// Admin key from .env (passed via build args)
const CONVEX_ADMIN_KEY = import.meta.env.VITE_CONVEX_ADMIN_KEY || ''

function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={handleCopy}
      className="shrink-0"
      title={`Copy ${label}`}
    >
      {copied ? (
        <Check className="h-4 w-4 text-green-500" />
      ) : (
        <Copy className="h-4 w-4" />
      )}
    </Button>
  )
}

function AdminKeyInput({ value }: { value: string }) {
  const [visible, setVisible] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  if (!value) {
    return (
      <div className="flex gap-2">
        <Input
          value="Not configured - see .env file"
          readOnly
          className="font-mono text-sm bg-muted text-muted-foreground"
        />
      </div>
    )
  }

  return (
    <div className="flex gap-2">
      <Input
        type={visible ? 'text' : 'password'}
        value={value}
        readOnly
        className="font-mono text-sm bg-muted"
      />
      <Button
        variant="outline"
        size="icon"
        onClick={() => setVisible(!visible)}
        className="shrink-0"
        title={visible ? 'Hide' : 'Show'}
      >
        {visible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
      </Button>
      <Button
        variant="outline"
        size="icon"
        onClick={handleCopy}
        className="shrink-0"
        title="Copy Admin Key"
      >
        {copied ? (
          <Check className="h-4 w-4 text-green-500" />
        ) : (
          <Copy className="h-4 w-4" />
        )}
      </Button>
    </div>
  )
}

function ConvexPage() {
  return (
    <div className="flex flex-col gap-6 p-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Convex Dashboard</h1>
        <p className="text-muted-foreground">
          Self-hosted backend configuration and access
        </p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Server className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Connection Details</CardTitle>
          </div>
          <CardDescription>
            Credentials for the self-hosted Convex backend used for real-time data synchronization
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="deployment-url" className="flex items-center gap-2">
              <Server className="h-3.5 w-3.5 text-muted-foreground" />
              Deployment URL
            </Label>
            <div className="flex gap-2">
              <Input
                id="deployment-url"
                value={CONVEX_URL}
                readOnly
                className="font-mono text-sm bg-muted"
              />
              <CopyButton value={CONVEX_URL} label="Deployment URL" />
            </div>
            <p className="text-xs text-muted-foreground">
              The URL where your Convex backend is running
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="admin-key" className="flex items-center gap-2">
              <Key className="h-3.5 w-3.5 text-muted-foreground" />
              Admin Key
            </Label>
            <AdminKeyInput value={CONVEX_ADMIN_KEY} />
            <p className="text-xs text-muted-foreground">
              {CONVEX_ADMIN_KEY
                ? 'Your admin key from .env - use this to authenticate with the dashboard'
                : 'Admin key not found. Make sure CONVEX_ADMIN_KEY is set in your .env file and rebuild the web container.'}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <ExternalLink className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Quick Access</CardTitle>
          </div>
          <CardDescription>
            Inspect your database, view logs, and debug issues
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-md border bg-muted/30 p-3 text-sm text-muted-foreground space-y-1">
            <p>Browse and edit data in all tables (jobs, settings, datasets)</p>
            <p>View real-time function logs and errors</p>
            <p>Run queries and mutations directly</p>
          </div>

          <Button
            className="w-full"
            onClick={() => window.open(CONVEX_DASHBOARD_URL, '_blank')}
          >
            <ExternalLink className="mr-2 h-4 w-4" />
            Open Convex Dashboard
          </Button>

          <p className="text-xs text-center text-muted-foreground">
            Opens at <code className="bg-muted px-1 rounded text-[11px]">{CONVEX_DASHBOARD_URL}</code>
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Troubleshooting</CardTitle>
          </div>
          <CardDescription>
            Common issues and solutions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div>
              <p className="font-medium">Dashboard won't load?</p>
              <p className="text-muted-foreground">
                Check container: <code className="bg-muted px-1 rounded text-[11px]">docker ps | grep convex</code>
              </p>
            </div>
            <div>
              <p className="font-medium">Invalid admin key?</p>
              <p className="text-muted-foreground">
                Regenerate: <code className="bg-muted px-1 rounded text-[11px]">docker exec convex ./generate_admin_key.sh</code>
              </p>
            </div>
            <div>
              <p className="font-medium">Admin key not showing?</p>
              <p className="text-muted-foreground">
                Add to <code className="bg-muted px-1 rounded text-[11px]">.env</code> and rebuild: <code className="bg-muted px-1 rounded text-[11px]">docker compose up --build web</code>
              </p>
            </div>
            <div>
              <p className="font-medium">Connection refused?</p>
              <p className="text-muted-foreground">
                Ensure ports 3210 (backend) and 6791 (dashboard) are exposed in Docker.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
