import { createFileRoute, useNavigate, useSearch } from '@tanstack/react-router'
import { ChevronLeft } from 'lucide-react'
import { Button } from '../components/ui/button'
import { RqSelector } from '../components/research-tools/rq-selector'
import { Rq1Page } from '../components/research-tools/rq1/rq1-page'
import { Rq2Skeleton } from '../components/research-tools/rq2-skeleton'
import { Rq3Skeleton } from '../components/research-tools/rq3-skeleton'
import { Rq4Skeleton } from '../components/research-tools/rq4-skeleton'
import { RQ_DEFINITIONS } from '../components/research-tools/types'
import type { RqId } from '../components/research-tools/types'
import { z } from 'zod'

const rqValues = ['rq1', 'rq2', 'rq3', 'rq4'] as const

const searchSchema = z.object({
  rq: z.enum(rqValues).optional(),
  table: z.string().optional(),
})

export const Route = createFileRoute('/research-tools')({
  component: ResearchToolsPage,
  validateSearch: searchSchema,
})

function ResearchToolsPage() {
  const navigate = useNavigate()
  const { rq: selectedRq, table: selectedTable } = useSearch({ from: '/research-tools' })

  const handleRqSelect = (rqId: RqId) => {
    navigate({ to: '/research-tools', search: { rq: rqId, table: undefined } })
  }

  const handleBack = () => {
    navigate({ to: '/research-tools', search: { rq: undefined, table: undefined } })
  }

  const handleTableChange = (table: string) => {
    navigate({ to: '/research-tools', search: { rq: selectedRq, table }, replace: true })
  }

  const rqDef = selectedRq ? RQ_DEFINITIONS.find(r => r.id === selectedRq) : null

  return (
    <div className="flex flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Research Tools</h1>
        <p className="text-muted-foreground">
          Generate paper-ready tables for thesis research questions
        </p>
      </div>

      {selectedRq ? (
        <div className="space-y-4">
          {/* Back button + RQ title */}
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={handleBack}>
              <ChevronLeft className="h-4 w-4 mr-1" />
              Back
            </Button>
            {rqDef && (
              <div>
                <h2 className="text-lg font-semibold">{rqDef.title}</h2>
                <p className="text-sm text-muted-foreground">{rqDef.description}</p>
              </div>
            )}
          </div>

          {/* RQ content */}
          {selectedRq === 'rq1' && (
            <Rq1Page
              currentTable={selectedTable || '1a'}
              onTableChange={handleTableChange}
            />
          )}
          {selectedRq === 'rq2' && <Rq2Skeleton />}
          {selectedRq === 'rq3' && <Rq3Skeleton />}
          {selectedRq === 'rq4' && <Rq4Skeleton />}
        </div>
      ) : (
        <RqSelector onSelect={handleRqSelect} />
      )}
    </div>
  )
}
