use std::fmt::Display;
use std::ops::IndexMut;
use bincode;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use strum::IntoEnumIterator;
use sqlx::Type;
use time::OffsetDateTime;
use uuid::Uuid;
use strum_macros::EnumIter;

#[derive(Deserialize, Serialize, Encode, Decode, Type, Clone, EnumIter, Debug)]
pub enum FieldName {
    Text,
    TokenIds,
    Logits,
    UUID,
    Timestamp
}

impl Display for FieldName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            FieldName::Text => {
                "text".to_string()
            }
            FieldName::TokenIds => {
                "token_ids".to_string()
            }
            FieldName::UUID => {
                "uuid".to_string()
            }
            FieldName::Logits => {
                "logits".to_string()
            }
            FieldName::Timestamp => {
                "timestamp".to_string()
            }
        };
        write!(f, "{}", str)
    }
}



pub type PyModelOutput = (String, String, String, Vec<u32>, Vec<f32>);

#[derive(Deserialize, Serialize, Encode, Decode, Type, Clone, Debug)]
pub struct Field<T: Clone + bincode::Encode> {
    pub field_name: FieldName,
    pub value: T,
}

impl<T: Clone + bincode::Encode + bincode::Decode<()>> Field<T> {
    pub fn encode(&self) -> Result<Vec<u8>, sqlx::Error> {
        bincode::encode_to_vec(self.value.clone(), bincode::config::standard())
            .map_err(|e| sqlx::Error::Decode(sqlx::error::BoxDynError::from(e)))
    }

    pub fn decode(data: &[u8]) -> Result<T, DecodeError> {
        let (result, _) = bincode::decode_from_slice(data, bincode::config::standard())?;
        Ok(result)
    }
}

#[derive(Serialize, Deserialize, Encode, Decode, Clone)]
pub struct ModelOutput {
    pub uuid: Field<String>,
    pub timestamp: Field<String>,
    pub text: Field<String>,
    pub token_ids: Field<Vec<u32>>,
    pub logits: Field<Vec<f32>>,
}

impl ModelOutput {
    pub fn new(text: String, token_ids: Vec<u32>, logits: Vec<f32>) -> Self {
        let output_text = Field {
            field_name: FieldName::Text,
            value: text,
        };
        let output_token_ids = Field {
            field_name: FieldName::TokenIds,
            value: token_ids,
        };
        let output_logits = Field {
            field_name: FieldName::Logits,
            value: logits,
        };
        let output_uuid = Field {
            field_name: FieldName::UUID,
            value: Uuid::new_v4().to_string(),
        };
        let output_time = Field {
            field_name: FieldName::Timestamp,
            value: OffsetDateTime::now_utc().to_string(),
        };

        ModelOutput {
            uuid: output_uuid,
            timestamp: output_time,
            text: output_text,
            token_ids: output_token_ids,
            logits: output_logits,
        }
    }

    pub fn serialize(&self) -> Result<Vec<u8>, EncodeError> {
        bincode::encode_to_vec(self, bincode::config::standard())
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, DecodeError> {
        let (result, _) = bincode::decode_from_slice(data, bincode::config::standard())?;
        Ok(result)
    }

    pub fn as_py_tuple(&self) -> PyModelOutput {
        (
            self.uuid.value.to_string(),
            self.timestamp.value.to_string(),
            self.text.value.to_string(),
            self.token_ids.value.clone(),
            self.logits.value.clone(),
        )
    }

    pub fn generate_table_query() -> String {
        let mut str_builder = vec![];
        str_builder.push("CREATE TABLE IF NOT EXISTS model_outputs (\n".to_string());
        str_builder.push("\tid INTEGER PRIMARY KEY,\n".to_string());
        for field in FieldName::iter() {
            str_builder.push(format!("\t{} TEXT NOT NULL,\n", field.to_string()));
        }
        if let Some(last) = str_builder.last_mut() {
            *last = last.replace("NULL,", "NULL");
        }
        str_builder.push("\t)".to_string());
        str_builder.join("")
    }

    pub fn get_top_token_id(&self, top_k: u64) -> Vec<(u32, f32)> {
        let mut top_vec = vec![];
        let mut logits = self.logits.value.clone();

        logits.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let tops = logits.iter().take(top_k as usize);
        for top in tops {
            let max_val_pos = self.logits.value.iter().position(|x| x == top).unwrap();
            let token_id = self.token_ids.value[max_val_pos];
            top_vec.push((token_id, top.clone()))
        }
        top_vec
    }
}
