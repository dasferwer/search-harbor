import asyncio
import json
import os
from pathlib import Path

from .catalog import canonical
from .db import engine, execute, one


async def seed(variants=None):
    variants = variants or int(os.getenv("CATALOG_VARIANTS", "100"))
    if not 1 <= variants <= 2000:
        raise ValueError("Choose 1 to 2000 variants per family")
    products = json.loads(Path("data/catalog.json").read_text())
    async with engine.begin() as conn:
        state = await one(conn, "SELECT * FROM catalog_state WHERE id=1 FOR UPDATE")
        sequence = state["sequence"]
        for product in products:
            for variant in range(variants):
                sku = f"{product['family']}-{variant:04}"
                if await one(conn, "SELECT sku FROM products WHERE sku=:sku", sku=sku):
                    continue
                sequence += 1
                body = {
                    **product,
                    "sku": sku,
                    "version": sequence,
                    "deleted": False,
                    "in_stock": variant % 7 != 0,
                    "price_cents": product["price_cents"] + variant * 10,
                }
                await execute(
                    conn,
                    "INSERT INTO products(sku,version,body) VALUES(:sku,:version,CAST(:body AS jsonb))",
                    sku=sku,
                    version=sequence,
                    body=canonical(body),
                )
                await execute(
                    conn,
                    "INSERT INTO events(sequence,sku,body) VALUES(:seq,:sku,CAST(:body AS jsonb))",
                    seq=sequence,
                    sku=sku,
                    body=canonical(body),
                )
        await execute(
            conn, "UPDATE catalog_state SET sequence=:sequence WHERE id=1", sequence=sequence
        )
        if not await one(conn, "SELECT name FROM generations LIMIT 1"):
            await execute(
                conn, "INSERT INTO generations(name,status) VALUES('catalog-initial','building')"
            )
    return sequence


if __name__ == "__main__":
    print(asyncio.run(seed()))
