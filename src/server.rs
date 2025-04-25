use axum::{
    extract::{Json, Path, Query},
    http::{header, StatusCode},
    body::Body,
    routing::get,
    response::Response,
    Router,
};
use axum::response::IntoResponse;
use serde_json::{Value, json};
use serde::Deserialize;
use crate::db::query_field;
use crate::outputs::{Field, FieldName, ModelOutput};


#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(handle_json));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[derive(Deserialize)]
pub struct GetRequest {
    pub text: String,
    pub top_k: Option<u64>,
}

pub async fn remove_leading_and_trailing_quotes(text: String) -> String {
    let replaced = text.replace("\"", "");
    replaced
}

pub async fn handle_json(Query(params): Query<GetRequest>) -> Result<Response, StatusCode> {
    let parsed_text = remove_leading_and_trailing_quotes(params.text.clone()).await;
    let output: ModelOutput = query_field(Field {
        field_name: FieldName::Text,
        value: parsed_text,
    })
        .await
        .unwrap();

    let response = if let Some(top_k) = params.top_k {
        dbg!("Top k is ", top_k);
        let top_tokens = output.get_top_token_ids(top_k);
        json!({
            "text": params.text,
            "logits": output.logits.value,
            "token_ids": output.token_ids.value,
            "top_k": top_tokens
        })
    } else {
        json!({
            "text": params.text,
            "logits": output.logits.value,
            "token_ids": output.token_ids.value
        })
    };

    Ok(Json(response).into_response())
}
