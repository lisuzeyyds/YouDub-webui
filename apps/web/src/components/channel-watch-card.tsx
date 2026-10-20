"use client"

import { useCallback, useEffect, useState } from "react"
import { RefreshCw, Trash2 } from "lucide-react"

import {
  addChannel,
  ApiError,
  checkChannels,
  getChannels,
  removeChannel,
  setChannelEnabled,
  type ChannelsResponse,
} from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

function fill(template: string, values: Record<string, string | number>) {
  return Object.entries(values).reduce(
    (text, [key, value]) => text.replace(`{${key}}`, String(value)),
    template,
  )
}

function seenLabel(value: string, never: string) {
  if (!value) return never
  const stamp = new Date(value)
  if (Number.isNaN(stamp.getTime())) return value
  return stamp.toLocaleString()
}

export function ChannelWatchCard() {
  const { t } = useI18n()
  const [data, setData] = useState<ChannelsResponse | null>(null)
  const [url, setUrl] = useState("")
  const [busy, setBusy] = useState<"add" | "check" | "">("")
  const [message, setMessage] = useState("")

  const reload = useCallback(async () => {
    try {
      setData(await getChannels())
    } catch (error) {
      setMessage(error instanceof ApiError ? error.message : String(error))
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  async function handleAdd() {
    const value = url.trim()
    if (!value || busy) return
    setBusy("add")
    setMessage("")
    try {
      await addChannel(value)
      setUrl("")
      await reload()
    } catch (error) {
      setMessage(error instanceof ApiError ? error.message : String(error))
    } finally {
      setBusy("")
    }
  }

  async function handleCheck() {
    setBusy("check")
    setMessage("")
    try {
      const result = await checkChannels()
      setMessage(
        fill(t.settings.channelsChecked, {
          channels: result.channels,
          queued: result.queued,
        }),
      )
      await reload()
    } catch (error) {
      setMessage(error instanceof ApiError ? error.message : String(error))
    } finally {
      setBusy("")
    }
  }

  const channels = data?.channels ?? []

  return (
    <div className="grid gap-2 rounded-lg border border-border/60 p-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <p className="text-sm font-medium">{t.settings.channelsSection}</p>
          <p className="text-xs text-muted-foreground">{t.settings.channelsHelp}</p>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => void handleCheck()}
          disabled={busy !== "" || channels.length === 0}
        >
          <RefreshCw className={`mr-1 h-3.5 w-3.5 ${busy === "check" ? "animate-spin" : ""}`} />
          {busy === "check" ? t.settings.channelsChecking : t.settings.channelsCheck}
        </Button>
      </div>

      <div className="flex gap-2">
        <Input
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              // The dialog wraps everything in a save form; Enter here must add a
              // channel, not submit the settings form.
              event.preventDefault()
              void handleAdd()
            }
          }}
          placeholder={t.settings.channelsPlaceholder}
          className="text-xs"
        />
        <Button type="button" size="sm" onClick={() => void handleAdd()} disabled={busy !== "" || !url.trim()}>
          {t.settings.channelsAdd}
        </Button>
      </div>

      {channels.length === 0 ? (
        <p className="text-xs text-muted-foreground">{t.settings.channelsEmpty}</p>
      ) : (
        <ul className="divide-y divide-border/60">
          {channels.map((channel) => (
            <li key={channel.channel_id} className="flex items-center justify-between gap-2 py-2">
              <div className="min-w-0">
                <p className="truncate text-xs font-medium">{channel.title || channel.channel_id}</p>
                <p className="truncate text-[11px] text-muted-foreground">
                  {t.settings.channelsLastSeen} {seenLabel(channel.last_seen, t.settings.channelsNever)}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-[11px]"
                  onClick={() => void setChannelEnabled(channel.channel_id, !channel.enabled).then(reload)}
                >
                  {channel.enabled ? "ON" : "OFF"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label={t.settings.channelsRemove}
                  className="h-7 w-7"
                  onClick={() => void removeChannel(channel.channel_id).then(reload)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {data?.status ? (
        <p className="text-[11px] text-muted-foreground">
          {fill(t.settings.channelsInterval, { minutes: data.status.interval_minutes })}
          {data.status.last_poll ? ` · ${seenLabel(data.status.last_poll, "")}` : ""}
          {data.status.last_error ? ` · ${t.settings.channelsError}: ${data.status.last_error}` : ""}
        </p>
      ) : null}

      {message ? (
        <p className="text-xs" aria-live="polite">
          {message}
        </p>
      ) : null}
    </div>
  )
}
