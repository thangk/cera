import { useState, useCallback } from 'react'
import { Upload, FileText } from 'lucide-react'
import { Badge } from './badge'

interface FileDropZoneProps {
  onFile?: (file: File) => void
  onFileSelect?: (file: File | null) => void  // Alternative callback that supports null (for clearing)
  file?: File | null
  selectedFileName?: string | null  // Alternative: just display the filename
  accept?: string
  formatLabel?: string
  placeholder?: string
  description?: string
  className?: string
}

export function FileDropZone({
  onFile,
  onFileSelect,
  file,
  selectedFileName,
  accept = '.jsonl,.csv,.xml',
  formatLabel,
  placeholder = 'Drop a file here, or click to browse',
  description,
  className = '',
}: FileDropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)

  // Use either callback
  const handleFile = useCallback((f: File) => {
    if (onFile) onFile(f)
    if (onFileSelect) onFileSelect(f)
  }, [onFile, onFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const handleClick = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = accept
    input.onchange = (e) => {
      const f = (e.target as HTMLInputElement).files?.[0]
      if (f) handleFile(f)
    }
    input.click()
  }, [accept, handleFile])

  // Display name: prefer file.name, fall back to selectedFileName
  const displayName = file?.name || selectedFileName

  return (
    <div
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`border-2 border-dashed rounded-lg px-5 py-8 text-center transition-colors cursor-pointer ${
        isDragOver ? 'border-primary bg-primary/10' : 'hover:border-primary/50'
      } ${className}`}
    >
      {displayName ? (
        <div className="flex items-center justify-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">{displayName}</span>
          {formatLabel && <Badge variant="secondary" className="text-[10px]">{formatLabel}</Badge>}
        </div>
      ) : (
        <div className="flex flex-col items-center gap-1.5">
          <Upload className="h-6 w-6 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">{placeholder}</p>
          {description && <p className="text-xs text-muted-foreground">{description}</p>}
        </div>
      )}
    </div>
  )
}
