use bincode;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use sqlx::Type;
use time::OffsetDateTime;
use uuid::Uuid;

pub static DB_TEXT_FIELD_NAME: &str = "text";
pub static DB_TOKEN_IDS_FIELD_NAME: &str = "token_ids";
pub static DB_LOGITS_FIELD_NAME: &str = "logits";
pub static DB_UUID_FIELD_NAME: &str = "uuid";
pub static DB_TIME_FIELD_NAME: &str = "timestamp";

pub type PyModelOutput = (String, String, String, Vec<u32>, Vec<f32>);

#[derive(Deserialize, Serialize, Encode, Decode, Type, Clone, Debug)]
pub struct Field<T: Clone + bincode::Encode> {
    pub field_name: String,
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
            field_name: DB_TEXT_FIELD_NAME.to_string(),
            value: text,
        };
        let output_token_ids = Field {
            field_name: DB_TOKEN_IDS_FIELD_NAME.to_string(),
            value: token_ids,
        };
        let output_logits = Field {
            field_name: DB_LOGITS_FIELD_NAME.to_string(),
            value: logits,
        };
        let output_uuid = Field {
            field_name: DB_UUID_FIELD_NAME.to_string(),
            value: Uuid::new_v4().to_string(),
        };
        let output_time = Field {
            field_name: DB_TIME_FIELD_NAME.to_string(),
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
}
