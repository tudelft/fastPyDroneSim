"""
    Opens a websocket server and provides a method to stream json pose data

    Copyright (C) 2024 Till Blaha -- TU Delft

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import asyncio
import websockets
import json
import numpy as np

class dummyInterface:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        pass

class wsInterface:
    def __init__(self, port):
        self.ws = None
        self.port = port 

    async def __aenter__(self):
        self.coro = self.start_server()
        asyncio.create_task(self.coro)

        await asyncio.sleep(2)

        return self

    async def __aexit__(self, *args, **kwargs):
        pass
        #await self.coro.__aexit__()

    async def handle_client(self, websocket, path):
        # Send data to the client
        self.ws = websocket
        while True:
            await asyncio.Future() # run forever without blokcing?

    async def start_server(self):
        # Start WebSocket server
        async with websockets.serve(self.handle_client, "localhost", self.port):
            await asyncio.Future()  # run forever

    async def sendData(self, xs):
        for j, xD in enumerate(xs):
            # plot a maximum of Nviz
            if not np.isnan(xD).any():
                data = {"id": j, "pos": list(xD[:3]), "quat": list(xD[6:10])}
                await self.ws.send(json.dumps(data))
