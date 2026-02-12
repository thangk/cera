import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { RQ_DEFINITIONS } from './types'

const rq2Def = RQ_DEFINITIONS.find(r => r.id === 'rq2')!

export function Rq2Skeleton() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{rq2Def.title}</CardTitle>
        <CardDescription>{rq2Def.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">
          This RQ will include {rq2Def.tables.length} tables for ablation study analysis.
          Implementation coming in a follow-up session.
        </p>
        <div className="flex flex-wrap gap-2">
          {rq2Def.tables.map(table => (
            <Badge key={table.id} variant="outline">
              {table.id}: {table.label}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
