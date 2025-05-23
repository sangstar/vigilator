use crate::outputs::{Field, ModelOutput, FieldName};
use once_cell::sync::{Lazy, OnceCell};
use pyo3::{PyErr, Python};
use sqlx::{Row, SqlitePool};
use tokio;
use tokio::runtime::Runtime;

pub static TOKIO_RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("Failed to create tokio runtime"));

pub static SQLITE_POOL: OnceCell<SqlitePool> = OnceCell::new();

fn retrieve_metadata_from_row(row: &sqlx::sqlite::SqliteRow) -> Result<ModelOutput, PyErr> {
    let uuid: String = row.try_get(&*FieldName::UUID.to_string()).expect("Failed to get uuid");
    let timestamp: String = row.try_get(&*FieldName::Timestamp.to_string()).expect("Failed to get timestamp");
    let text: String = row.try_get(&*FieldName::Text.to_string()).expect("Failed to get text");
    let token_ids: Vec<u8> = row.try_get(&*FieldName::TokenIds.to_string()).expect("Failed to get token_ids");
    let logits: Vec<u8> = row.try_get(&*FieldName::Logits.to_string()).expect("Failed to get logits");

    let token_id_decoded = Field::<Vec<u32>>::decode(&token_ids).map_err(|e| {
        Python::with_gil(|py| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to decode: {}", e))
        })
    })?;

    let logits_decoded = Field::<Vec<f32>>::decode(&logits).map_err(|e| {
        Python::with_gil(|py| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to decode: {}", e))
        })
    })?;

    Ok(ModelOutput {
        uuid: Field {
            field_name: FieldName::UUID,
            value: uuid,
        },
        timestamp: Field {
            field_name: FieldName::Timestamp,
            value: timestamp,
        },
        text: Field {
            field_name: FieldName::Text,
            value: text,
        },
        token_ids: Field {
            field_name: FieldName::TokenIds,
            value: token_id_decoded,
        },
        logits: Field {
            field_name: FieldName::Logits,
            value: logits_decoded,
        },
    })
}

async fn _debug_table_contents(pool: &SqlitePool) {
    let rows = sqlx::query(
        r#"
        SELECT *
        FROM model_outputs
        "#,
    )
    .fetch_all(pool)
    .await
    .expect("Failed to fetch table contents");

    for row in rows {
        let id: i32 = row.try_get("id").expect("Failed to get id");
        let uuid: String = row.try_get(&*FieldName::UUID.to_string()).expect("Failed to get uuid");
        let timestamp: String = row.try_get(&*FieldName::Timestamp.to_string()).expect("Failed to get timestamp");
        let text: String = row.try_get(&*FieldName::Text.to_string()).expect("Failed to get text");
        let token_ids: Vec<u8> = row.try_get(&*FieldName::TokenIds.to_string()).expect("Failed to get token_ids");
        let logits: Vec<u8> = row.try_get(&*FieldName::Logits.to_string()).expect("Failed to get logits");

        dbg!(
            id,
            uuid,
            timestamp,
            text,
            Field::<Vec<u32>>::decode(&token_ids)
                .unwrap_or_else(|e| { panic!("Failed to decode token_ids: {:?}", e) }),
            Field::<Vec<f32>>::decode(&logits)
                .unwrap_or_else(|e| { panic!("Failed to decode logits: {:?}", e) })
        );
    }
}

pub async fn debug_table_contents() {
    maybe_init_pool().await;
    _debug_table_contents(SQLITE_POOL.get().unwrap()).await;
}

async fn maybe_init_pool() {
    let pool = SqlitePool::connect("sqlite::memory:?cache=shared")
        .await
        .expect("Failed to connect");
    if SQLITE_POOL.set(pool).is_err() {
        return;
    }
}

async fn maybe_create_table(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    let raw_query = ModelOutput::generate_table_query();
    dbg!(&raw_query);
    sqlx::query(&raw_query)
    .execute(pool)
    .await?;
    Ok(())
}

async fn maybe_insert_output(pool: &SqlitePool, output: &ModelOutput) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO model_outputs (uuid, timestamp, text, token_ids, logits)
        VALUES (?, ?, ?, ?, ?)
        "#,
    )
    .bind(&output.uuid.value)
    .bind(&output.timestamp.value)
    .bind(&output.text.value)
    .bind(&output.token_ids.encode()?)
    .bind(&output.logits.encode()?)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn insert_output(output: &ModelOutput) -> Result<(), sqlx::Error> {
    maybe_init_pool().await;
    let pool = &SQLITE_POOL.get().unwrap();
    maybe_create_table(pool).await?;
    maybe_insert_output(pool, &output).await?;
    Ok(())
}

pub async fn query_field<T>(field: Field<T>) -> Result<ModelOutput, PyErr>
where
    T: for<'q> sqlx::Encode<'q, sqlx::Sqlite> // T must be encodable in to the DB
        + sqlx::Type<sqlx::Sqlite> // T must be a recognized type by DB
        + bincode::Encode // T must be able to be encoded by `bincode` in to binary data
        + Clone // T must be cloneable
        + bincode::Decode<()>
        // T must be able to be decoded from the binary data to a Field
        + std::fmt::Debug,
{
    maybe_init_pool().await;
    debug_table_contents().await;
    let pool = &SQLITE_POOL.get().unwrap();
    let query = format!(
        r#"
        SELECT *
        FROM model_outputs
        WHERE {} = (?)
        "#,
        field.field_name
    );

    dbg!(query.clone());
    dbg!(field.value.clone());

    let row = sqlx::query(&query)
        .bind(field.value)
        .fetch_one(*pool)
        .await
        .map_err(|_e| {
            Python::with_gil(|py| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to retrieve query.")
            })
        })?;

    retrieve_metadata_from_row(&row)
}
