use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::num::Wrapping;

type Timestamp = u64;
type TimestampDelta = i32;
type Value = u64;
type Position = u8;

const BLOCK_SIZE: u8 = 64;
const VALUE_SIZE: u8 = 64;
const INITIAL_TS_SIZE: u8 = 14;
const INITIAL_VALUE_SIZE: u8 = 64;

#[derive(Debug)]
struct SeriesEntry {
    encoding: Vec<u64>,
    insert_pos: Position,

    prev_ts: Timestamp,
    prev_ts_delta: TimestampDelta,

    prev_value: Value,
    prev_num_leading_zeros: u8,
    prev_num_trailing_zeros: u8,
}

#[derive(Debug)]
struct DataPoint {
    timestamp: Timestamp,
    value: Value,
}

#[derive(Debug)]
struct Series {
    entries: HashMap<Timestamp, SeriesEntry>
}

#[derive(Debug)]
struct DecodeState {
    block_pos: Position,
    bit_pos: Position,

    prev_ts: Timestamp,
    prev_ts_delta: TimestampDelta,

    prev_value: Value,
    prev_num_leading_zeros: u8,
    prev_num_trailing_zeros: u8,
}

impl DecodeState {
    fn advance_pos(&mut self, inc: u8) {
        if self.bit_pos + inc >= BLOCK_SIZE {
            self.block_pos += 1;
            self.bit_pos = (self.bit_pos + inc) % BLOCK_SIZE;
        } else {
            self.bit_pos += inc;
        }
    }
}

impl SeriesEntry {
    fn show(&self) {
        println!(
            "insert_pos = {:?}, prev_ts = {:?}, prev_ts_delta = {:?}, prev_value = {:?},\
            prev_leading_zeros = {:?}, prev_trailing_zeros = {:?}",
            self.insert_pos, self.prev_ts, self.prev_ts_delta, self.prev_value,
            self.prev_num_leading_zeros, self.prev_num_trailing_zeros
        );

        println!("num encoding blocks = {:?}", self.encoding.len());

        for block in &self.encoding {
            println!("{:X}", block);
        }
    }

    fn encode_timestamp(&mut self, timestamp: Timestamp) {
        let curr_ts_delta: TimestampDelta =
            (timestamp.wrapping_sub(self.prev_ts)) as TimestampDelta;

        let ts_delta_of_delta = curr_ts_delta - self.prev_ts_delta;

        match ts_delta_of_delta {
            0 => {
                self.append_number(0b0, 1)
            },

            -63 ... 64 => {
                self.append_number(0b10, 2);
                self.append_number(ts_delta_of_delta as u64, 7);
            },

            -255 ... 256 => {
                self.append_number(0b110, 3);
                self.append_number(ts_delta_of_delta as u64, 9);
            }

            -2047 ... 2048 => {
                self.append_number(0b1110, 4);
                self.append_number(ts_delta_of_delta as u64, 12);
            }

            _ => {
                self.append_number(0b1111, 4);
                self.append_number(ts_delta_of_delta as u64, 32);
            }
        };

        self.prev_ts = timestamp;
        self.prev_ts_delta = curr_ts_delta;
    }

    pub fn encode_value(&mut self, value: Value) {
        let xor = value ^ self.prev_value;

        let (num_leading_zeros, num_trailing_zeros) =
            get_leading_and_trailing_zeros(xor);

        if xor == 0 {
            self.append_number(0b0, 1);
        } else {
            self.append_number(0b1, 1);

            let prev_num_leading_zeros = self.prev_num_leading_zeros;
            let prev_num_trailing_zeros = self.prev_num_trailing_zeros;

            // Does this fall within the previous block of meaningful bits?
            if num_leading_zeros >= prev_num_leading_zeros &&
                num_trailing_zeros >= prev_num_trailing_zeros {
                self.append_number(0b0, 1);
                self.append_number(
                    xor >> prev_num_trailing_zeros as u64,
                    VALUE_SIZE - (prev_num_leading_zeros + prev_num_trailing_zeros)
                );
            } else {
                self.append_number(0b1, 1);
                self.append_number(num_leading_zeros as u64, 6);
                let len_meaningful_bits = 64 - (num_leading_zeros + num_trailing_zeros);
                self.append_number(len_meaningful_bits as u64, 6);
                self.append_number(
                    xor >> num_trailing_zeros as u64,
                    len_meaningful_bits
                );
            }
        }

        self.prev_value = value;
        self.prev_num_leading_zeros = num_leading_zeros;
        self.prev_num_trailing_zeros = num_trailing_zeros;
    }

    pub fn append_number(&mut self, number: u64, size: u8) {
        // Have we used up the current block?
        if self.insert_pos >= BLOCK_SIZE {
            // If yes, then just append the number in a new block.
            self.encoding.push(number << (BLOCK_SIZE - size) as u64);
            self.insert_pos = size;
            return;
        }

        // Otherwise, we need to fill up the current block first.
        let curr_block_pos = self.encoding.len() - 1;
        let block = *self.encoding.last().unwrap();

        fn append_to_block(block: u64, insert_pos: Position, num: u64, num_size: u8) -> u64 {
            let offset = BLOCK_SIZE - insert_pos - num_size;
            let aligned_num = num << offset as u64;

            // Since this can be a negative number we're appending, discard
            // all unwanted 'FF..' padding left of the number.
            let discard_offset = insert_pos as u64;
            let appendable_number = (aligned_num << discard_offset) >> discard_offset;

            block | appendable_number
        }

        // Can the number fit in the current block?
        if self.insert_pos + size <= BLOCK_SIZE {
            // If yes, then we can just append it.
            let num_insert_pos = (BLOCK_SIZE - self.insert_pos - size) as u64;
            let curr_block = self.encoding[curr_block_pos];
            self.encoding[curr_block_pos] =
                append_to_block(curr_block, self.insert_pos, number, size);
            self.insert_pos += size;
            return;
        }

        // Otherwise, we use the first n bits of the number to fill up the
        // current block, add the remaining size - n bits as the new block.
        let num_bits_to_append_to_curr_block = BLOCK_SIZE - self.insert_pos;
        let num_bits_to_append_to_new_block = size - num_bits_to_append_to_curr_block;

        let start_pos_to_extract_first_word = BLOCK_SIZE - size;
        let start_pos_to_extract_second_word =
            start_pos_to_extract_first_word + num_bits_to_append_to_curr_block;

        let bits_to_append_to_first_word = SeriesEntry::extract_n_bits(
            number, start_pos_to_extract_first_word,
            num_bits_to_append_to_curr_block,
        );
        let bits_to_append_to_second_word = SeriesEntry::extract_n_bits(
            number, start_pos_to_extract_second_word,
            num_bits_to_append_to_new_block
        );

        // Append first bits to the current block.
        self.encoding[curr_block_pos] =
            block | (bits_to_append_to_first_word <<
                (BLOCK_SIZE - self.insert_pos - num_bits_to_append_to_curr_block) as u64);

        // Append the remaining bits bits to a new block.
        self.encoding.push(
            bits_to_append_to_second_word <<
                (BLOCK_SIZE - num_bits_to_append_to_new_block) as u64
        );
        self.insert_pos = num_bits_to_append_to_new_block;
    }

    fn extract_n_bits(word: u64, start_pos: Position, n: u8) -> u64 {
        if start_pos == 0 {
            ((std::u64::MAX << (BLOCK_SIZE - n)) & word) >> (BLOCK_SIZE - n)
        } else {
            (((1_u64 << (BLOCK_SIZE - start_pos) as u64) - 1) & word)
                >> (BLOCK_SIZE - (start_pos + n)) as u64
        }
    }

    fn decode(&self, window: Timestamp) -> Vec<DataPoint> {
        let mut points = Vec::new();

        let encoding = &self.encoding;

        if encoding.len() == 0 && self.insert_pos == 0 {
            return points;
        }

        let mut state = DecodeState{
            block_pos: 0,
            bit_pos: 0,

            prev_ts: window,
            prev_ts_delta: 0,

            prev_value: std::u64::MAX,
            prev_num_leading_zeros: 0,
            prev_num_trailing_zeros: 0,
        };

        let (initial_ts, initial_delta) =
            self.decode_initial_timestamp(&mut state);
        let initial_value = self.decode_initial_value(&mut state);

        points.push(DataPoint{ timestamp: initial_ts, value: initial_value });

        state.prev_ts = initial_ts;
        state.prev_ts_delta = initial_delta;

        state.prev_value = initial_value;
        let (leading, trailing) =
            get_leading_and_trailing_zeros(initial_value);
        state.prev_num_leading_zeros = leading;
        state.prev_num_trailing_zeros = trailing;

        let max_block_pos = (encoding.len() - 1) as Position;
        let max_bit_pos = self.insert_pos - 1;
        let max_pos = (encoding.len() - 1) as u64 * 64 + self.insert_pos as u64;

        while ((state.block_pos as u64 * 64) + state.bit_pos as u64) < max_pos {
            let timestamp = self.decode_timestamp(&mut state);
            let value = self.decode_value(&mut state);
            points.push(DataPoint{ timestamp, value });
        }

        points
    }

    fn decode_initial_timestamp(&self, state: &mut DecodeState)
        -> (Timestamp, TimestampDelta)
    {
        let initial_delta_of_delta = SeriesEntry::extract_n_bits(
            self.encoding[state.block_pos as usize],
            state.bit_pos, INITIAL_TS_SIZE
        ) as TimestampDelta;

        state.advance_pos(INITIAL_TS_SIZE);

        (
            get_timestamp_value(state.prev_ts, initial_delta_of_delta),
            initial_delta_of_delta
        )
    }

    fn decode_initial_value(&self, state: &mut DecodeState) -> Value {
        self.read_n_bits(state, INITIAL_VALUE_SIZE)
    }

    fn decode_timestamp(&self, state: &mut DecodeState) -> Timestamp {
        // Read the timestamp header.
        let mut num_ones = 0;
        while bit_is_set(self.encoding[state.block_pos as usize], state.bit_pos) {
            num_ones += 1;

            // Are we looking at the maximum case '1111'?
            if num_ones == 4 {
                break;
            }

            state.advance_pos(1);
        }

        // Advance past the last '0' (or 1 in the case of '1111') bit.
        state.advance_pos(1);

        // Read the delta of delta.
        let delta_of_delta = match num_ones {
            0 => 0,
            1 => self.read_delta_of_delta(state, 7),
            2 => self.read_delta_of_delta(state, 9),
            3 => self.read_delta_of_delta(state, 12),
            4 => self.read_delta_of_delta(state, 32),
            _ => panic!("Invalid encoding: Number of trailing ones in timestamp headers must be between [0,4]."),
        };

        let delta = delta_of_delta + state.prev_ts_delta;
        let timestamp = get_timestamp_value(state.prev_ts, delta);

        state.prev_ts = timestamp;
        state.prev_ts_delta = delta;

        timestamp
    }

    fn decode_value(&self, state: &mut DecodeState) -> Value {
        let (value, num_leading_zeros, num_trailing_zeros) =
        match self.read_n_bits(state, 1) {
            0 => (state.prev_value, state.prev_num_leading_zeros, state.prev_num_trailing_zeros),

            1 => {
                match self.read_n_bits(state, 1) {
                    0 => {
                        let num_leading_zeros = state.prev_num_leading_zeros;
                        let num_trailing_zeros = state.prev_num_trailing_zeros;
                        let len_meaningful_xor =
                            VALUE_SIZE - (num_leading_zeros + num_trailing_zeros);

                        let meaningful_xor = self.read_n_bits(state, len_meaningful_xor);
                        let xor = meaningful_xor << num_trailing_zeros as u64;
                        let value = state.prev_value ^ xor;

                        (value, state.prev_num_leading_zeros, state.prev_num_trailing_zeros)
                    },

                    1 => {
                        let num_leading_zeros = self.read_n_bits(state, 6) as u8;
                        let len_meaningful_xor = self.read_n_bits(state, 6) as u8;
                        let num_trailing_zeros =
                            VALUE_SIZE - (num_leading_zeros + len_meaningful_xor);

                        let meaningful_xor = self.read_n_bits(state, len_meaningful_xor);
                        let xor = meaningful_xor << num_trailing_zeros as u64;
                        let value = state.prev_value ^ xor;

                        (value, num_leading_zeros, num_trailing_zeros)
                    },

                    _ => panic!("Invalid encoding: control_bit for value must be 0 or 1")
                }
            },

            _ => panic!("Invalid encoding: xor_bit for value must be 0 or 1")
        };

        state.prev_value = value;
        state.prev_num_leading_zeros = num_leading_zeros;
        state.prev_num_trailing_zeros = num_trailing_zeros;

        value
    }

    fn read_delta_of_delta(&self, state: &mut DecodeState, size: u8) -> TimestampDelta
    {
        let bytes = self.read_n_bits(state, size);

        // Is this a negative delta?
        let delta_of_delta = if bit_is_set(bytes as u64, 64 - size) {
            // If yes, then convert the n-bit delta to it's <TimestampDelta> bit two's complement.
            ((std::u64::MAX << size) | bytes as u64) as TimestampDelta
        } else {
            bytes as TimestampDelta
        };

       delta_of_delta
    }

    fn read_n_bits(&self, state: &mut DecodeState, size: u8) -> u64 {
        // Are all requested bits in the same block?
        if state.bit_pos + size <= BLOCK_SIZE {
            // If yes, we can read it all in one go.
            let word = extract_n_bits(
                self.encoding[state.block_pos as usize],
                state.bit_pos, size
            );
            state.advance_pos(size);
            word
        } else {
            // Otherwise, we need to read the first n bits from the current block
            // and the remaining (size - n) bits from the next block.
            let num_bits_from_first_block = BLOCK_SIZE - state.bit_pos;
            let num_bits_from_second_block = size - num_bits_from_first_block;

            let bits_from_first_block = extract_n_bits(
                self.encoding[state.block_pos as usize],
                state.bit_pos, num_bits_from_first_block
            );

            let bits_from_second_block = extract_n_bits(
                self.encoding[state.block_pos as usize + 1],
                0, num_bits_from_second_block);

            // Merge both values into a single word.
            let word =
                (bits_from_first_block << num_bits_from_second_block as u64) |
                    bits_from_second_block;

            state.block_pos += 1;
            state.bit_pos = num_bits_from_second_block;

            word
        }
    }
}

impl Series {
    pub fn insert(&mut self, point: DataPoint) {
        let timestamp = point.timestamp;
        let value = point.value;

        let window = timestamp - (timestamp % (2 * 60 * 60));

        match  self.entries.entry(window) {
            Entry::Occupied(mut entry) => {
                let block = entry.get_mut();
                block.encode_timestamp(timestamp);
                block.encode_value(value);
            },

            Entry::Vacant(entry) => {
                // Store the initial delta in 14 bits.
                let delta = timestamp - window;

                let zeros = get_leading_and_trailing_zeros(value);
                let mut series_entry = SeriesEntry{
                    encoding: vec![0],
                    insert_pos: 0,

                    prev_ts: timestamp,
                    prev_ts_delta: delta as TimestampDelta,

                    prev_value: value,
                    prev_num_leading_zeros: zeros.0,
                    prev_num_trailing_zeros: zeros.1,
                };
                series_entry.append_number(delta, INITIAL_TS_SIZE);
                series_entry.append_number(value, 64);

                entry.insert(series_entry);
            }

        }
    }
}

fn get_timestamp_value(prev_ts: Timestamp, delta: TimestampDelta) -> Timestamp {
    // We can use signed integer operations here since the delta can be negative.
    // We don't expect any overflow during the cast back to Timestamp
    // since timestamps are never negative.
    (prev_ts as i64 + delta as i64) as Timestamp
}

fn get_leading_and_trailing_zeros(value: Value) -> (u8, u8) {
    let mut num_leading_zeros = 0;
    let mut num_trailing_zeros = 0;

    for bit_pos in 0..64 {
        if bit_is_set(value, bit_pos) {
            break;
        }
        num_leading_zeros += 1;
    }

    for bit_pos in (0..64).rev() {
        if bit_is_set(value, bit_pos) {
            break;
        }
        num_trailing_zeros += 1;
    }

    (num_leading_zeros, num_trailing_zeros)
}

fn extract_n_bits(word: u64, start_pos: Position, n: u8) -> u64 {
    if start_pos == 0 {
        ((std::u64::MAX << (BLOCK_SIZE - n)) & word) >> (BLOCK_SIZE - n)
    } else {
        (((1_u64 << (BLOCK_SIZE - start_pos) as u64) - 1) & word) >>
            (BLOCK_SIZE - (start_pos + n)) as u64
    }
}

fn bit_is_set(word: u64, bit_pos: Position) -> bool {
    (word & (1_u64 << (BLOCK_SIZE - bit_pos - 1) as u64)) != 0
}

#[cfg(test)]
mod tests {
    use ::{std, SeriesEntry};
    use std::num::Wrapping;
    use ::{Series, DataPoint};
    use std::collections::HashMap;
    use ::{get_leading_and_trailing_zeros, Timestamp};

    #[test]
    fn test_case() {
        let mut m = Series{ entries: HashMap::new() };

        let ts_base = 20000;
        let timestamps: Vec<Timestamp> = vec![0, 60, 62, 10, 20]
            .iter().map(|x| ts_base + *x).collect();
        let values = [10, 20, 30, 40, 50];

        for (i, timestamp) in timestamps.iter().enumerate() {
            m.insert(DataPoint{timestamp: *timestamp, value: values[i]});
        }

        for (window, entry) in &m.entries {
            let mut points = entry.decode(*window);
            for (i, point) in points.iter().enumerate() {
                assert_eq!(point.timestamp, timestamps[i]);
                assert_eq!(point.value, values[i]);
            }
        }
    }
}
