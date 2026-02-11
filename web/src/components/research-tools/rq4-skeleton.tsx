import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { RQ_DEFINITIONS } from './types'

const rq4Def = RQ_DEFINITIONS.find(r => r.id === 'rq4')!

export function Rq4Skeleton() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{rq4Def.title}</CardTitle>
        <CardDescription>{rq4Def.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">
          This RQ will include {rq4Def.tables.length} tables for cross-domain generalizability analysis.
          Implementation coming in a follow-up session.
        </p>
        <div className="flex flex-wrap gap-2">
          {rq4Def.tables.map(table => (
            <Badge key={table.id} variant="outline">
              {table.id}: {table.label}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
