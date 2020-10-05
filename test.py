import asyncio
async def coro(i):
    print("core start",i)
    ret = await get(i) # 此处为耗时的io等操作
    print("core end",i)
    return ret
async def get(i):
    await asyncio.sleep(5)
    return i
loop = asyncio.get_event_loop()
a = [asyncio.ensure_future(coro(i)) for i in range(10)]
loop.run_until_complete(asyncio.wait(a))
loop.close()
print([i.result() for i in a])
