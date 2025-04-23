use bincode;
use bincode::{Decode, Encode};
use bincode::error::{DecodeError, EncodeError};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct ModelOutput {
    pub token_ids: Vec<u32>,
    pub logits: Vec<f32>,
}

impl ModelOutput {
    pub fn new(token_ids: Vec<u32>, logits: Vec<f32>) -> Self {
        ModelOutput { token_ids, logits }
    }

    pub fn serialize(&self) -> Result<Vec<u8>, EncodeError> {
        bincode::encode_to_vec(self, bincode::config::standard())
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, DecodeError> {
        let (result, _) = bincode::decode_from_slice(data, bincode::config::standard())?;
        Ok(result)
    }
}
