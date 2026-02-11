import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Table1aMdqa } from './table-1a-mdqa'
import { Table1bLady } from './table-1b-lady'
import { Table1cStats } from './table-1c-stats'
import { RQ_DEFINITIONS } from '../types'

interface Rq1PageProps {
  currentTable: string
  onTableChange: (table: string) => void
}

const rq1Def = RQ_DEFINITIONS.find(r => r.id === 'rq1')!

export function Rq1Page({ currentTable, onTableChange }: Rq1PageProps) {
  const activeTable = currentTable || rq1Def.tables[0].id

  return (
    <div className="space-y-4">
      <Tabs value={activeTable} onValueChange={onTableChange}>
        <TabsList className="grid w-full grid-cols-3">
          {rq1Def.tables.map(table => (
            <TabsTrigger key={table.id} value={table.id}>
              {table.id}: {table.label}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="1a" className="mt-4">
          <Table1aMdqa />
        </TabsContent>

        <TabsContent value="1b" className="mt-4">
          <Table1bLady />
        </TabsContent>

        <TabsContent value="1c" className="mt-4">
          <Table1cStats />
        </TabsContent>
      </Tabs>
    </div>
  )
}
