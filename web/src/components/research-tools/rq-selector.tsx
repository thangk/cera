import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { FlaskConical } from 'lucide-react'
import { RQ_DEFINITIONS } from './types'
import type { RqId } from './types'

interface RqSelectorProps {
  onSelect: (rqId: RqId) => void
}

export function RqSelector({ onSelect }: RqSelectorProps) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {RQ_DEFINITIONS.map((rq) => (
        <Card
          key={rq.id}
          className="cursor-pointer transition-colors hover:bg-muted/50"
          onClick={() => onSelect(rq.id)}
        >
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <FlaskConical className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-base">{rq.title}</CardTitle>
            </div>
            <CardDescription>{rq.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1.5">
              {rq.tables.map((table) => (
                <Badge key={table.id} variant="secondary" className="text-xs">
                  {table.id}: {table.label}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
