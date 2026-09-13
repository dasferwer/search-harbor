from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=3, max_length=180)
    description: str = Field(min_length=10, max_length=2000)
    category: str = Field(min_length=1, max_length=60, pattern=r"^[a-z0-9_-]+$")
    brand: str = Field(min_length=1, max_length=60, pattern=r"^[a-z0-9_-]+$")
    family: str = Field(min_length=1, max_length=60, pattern=r"^[a-z0-9_-]+$")
    price_cents: int = Field(ge=0, le=100_000_000, strict=True)
    in_stock: bool = Field(strict=True)


class Search(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    mode: Literal["lexical", "semantic", "hybrid", "rerank"] = "hybrid"
    category: str | None = Field(default=None, max_length=60)
    brand: str | None = Field(default=None, max_length=60)
    min_price: int = Field(default=0, ge=0, le=100_000_000)
    max_price: int = Field(default=100_000_000, ge=0, le=100_000_000)
    in_stock: bool = True
    limit: int = Field(default=10, ge=1, le=50)
    distinct_families: bool = False

    @model_validator(mode="after")
    def valid_query(self):
        self.query = self.query.strip()
        if not self.query or self.min_price > self.max_price:
            raise ValueError("Query must not be blank and price range must be ordered")
        return self
