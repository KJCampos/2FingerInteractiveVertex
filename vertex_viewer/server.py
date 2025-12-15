import json
from aiohttp import web

clients = set()

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    clients.add(ws)
    print("Viewer connected:", len(clients))

    try:
        async for _ in ws:
            pass
    finally:
        clients.discard(ws)
        print("Viewer disconnected:", len(clients))

    return ws

async def _broadcast(data: dict):
    if not clients:
        return
    payload = json.dumps(data)
    dead = []
    for ws in clients:
        try:
            await ws.send_str(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)

async def post_vertex(request):
    data = await request.json()
    data["type"] = data.get("type", "vertex")
    print("RX /vertex:", data)
    await _broadcast(data)
    return web.json_response({"ok": True, "viewers": len(clients)})

async def post_hand(request):
    data = await request.json()
    data["type"] = data.get("type", "hand")
    await _broadcast(data)
    return web.json_response({"ok": True, "viewers": len(clients)})

app = web.Application()
app.router.add_get("/ws", ws_handler)
app.router.add_post("/vertex", post_vertex)
app.router.add_post("/hand", post_hand)

# If you already have a viewer file, keep your existing route.
# Example:
# app.router.add_static("/", path="web", show_index=True)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8765)
