import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { RQ_DEFINITIONS } from './types'

const rq3Def = RQ_DEFINITIONS.find(r => r.id === 'rq3')!

export function Rq3Skeleton() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{rq3Def.title}</CardTitle>
        <CardDescription>{rq3Def.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">
          This RQ will include {rq3Def.tables.length} tables for scaling analysis.
          Implementation coming in a follow-up session.
        </p>
        <div className="flex flex-wrap gap-2">
          {rq3Def.tables.map(table => (
            <Badge key={table.id} variant="outline">
              {table.id}: {table.label}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
