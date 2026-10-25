"""Server defaults for columns whose model default is a plain value.

A model field such as ``is_owner: bool = False`` gives SQLAlchemy only a
client-side default, so ``create_all`` makes the column without one. The
migration that once added it to an older database did give it one
(``ADD COLUMN … DEFAULT FALSE``), so a fresh database and an upgraded one
disagreed, and a row inserted by plain SQL failed NOT NULL only on the fresh
one. ``give_scalar_server_defaults`` puts the default on the column itself, so
``create_all`` emits it; ``apply_missing_server_defaults`` adds it to tables an
earlier ``create_all`` made without it.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

_INFO_KEY = "scalar_server_default"


def give_scalar_server_defaults(metadata: sa.MetaData) -> None:
    """Set a server default on each column whose default is a bool, number or string. Idempotent.

    A server default the model declares itself is recorded too, so that
    ``apply_missing_server_defaults`` brings an older table up to it.
    """
    dialect = postgresql.dialect()
    for table in metadata.tables.values():
        for column in table.columns:
            if _INFO_KEY in column.info:
                continue
            server_default = column.server_default
            if isinstance(server_default, sa.DefaultClause):
                arg = server_default.arg
                # A plain string is a value (SQLAlchemy quotes it), anything else is SQL.
                column.info[_INFO_KEY] = (
                    "'" + arg.replace("'", "''") + "'" if isinstance(arg, str) else str(arg.compile(dialect=dialect))
                )
                continue
            default = column.default
            if column.primary_key or server_default is not None or default is None:
                continue
            if getattr(default, "is_callable", False) and _is_timestamp(column):
                _set(column, "now()")
                continue
            if not getattr(default, "is_scalar", False):
                continue
            value = default.arg
            if not isinstance(value, (bool, int, float, str)):  # str Enums are str
                continue
            try:
                literal = str(sa.literal(value, type_=column.type).compile(
                    dialect=dialect, compile_kwargs={"literal_binds": True}
                ))
            except Exception:
                continue
            _set(column, literal)
    _from_model_fields()


def _is_timestamp(column: sa.Column) -> bool:
    # A timestamp made by a default_factory is "now" in every model here.
    return isinstance(column.type, sa.DateTime) and not column.nullable


def _set(column: sa.Column, literal: str) -> None:
    column.server_default = sa.DefaultClause(sa.text(literal))
    column.info[_INFO_KEY] = literal


def _from_model_fields() -> None:
    """The same for fields declared with ``sa_column=…``: SQLAlchemy never sees their
    model default, so read it from the SQLModel field (``default=`` a plain value,
    or ``default_factory=dict`` / ``list`` for a JSON column)."""
    from sqlmodel.main import default_registry

    for mapper in list(default_registry.mappers):
        table = getattr(mapper.class_, "__table__", None)
        fields = getattr(mapper.class_, "model_fields", None)
        if table is None or not fields:
            continue
        for name, field in fields.items():
            column = table.columns.get(name)
            if column is None or column.primary_key or column.nullable or column.server_default is not None:
                continue
            factory = getattr(field, "default_factory", None)
            if factory is dict or factory is list:
                literal = "'{}'" if factory is dict else "'[]'"
            elif factory is not None and _is_timestamp(column):
                literal = "now()"
            elif factory is None and isinstance(field.default, (bool, int, float, str)):
                try:
                    literal = str(sa.literal(field.default, type_=column.type).compile(
                        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
                    ))
                except Exception:
                    continue
            else:
                continue
            _set(column, literal)


async def apply_missing_server_defaults(conn, metadata: sa.MetaData) -> None:
    """ALTER the columns from ``give_scalar_server_defaults`` that have no default in the database.

    A database that already agrees takes no ALTER, and so no table lock, at startup.
    """
    wanted = {
        (table.name, column.name): column.info[_INFO_KEY]
        for table in metadata.tables.values()
        for column in table.columns
        if _INFO_KEY in column.info
    }
    if not wanted:
        return
    rows = await conn.execute(text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND column_default IS NULL"
    ))
    by_table: dict[str, list[str]] = {}
    for table_name, column_name in rows:
        literal = wanted.get((table_name, column_name))
        if literal is not None:
            by_table.setdefault(table_name, []).append(f'ALTER COLUMN "{column_name}" SET DEFAULT {literal}')
    for table_name, clauses in sorted(by_table.items()):
        await conn.execute(text(f'ALTER TABLE "{table_name}" ' + ", ".join(clauses)))
