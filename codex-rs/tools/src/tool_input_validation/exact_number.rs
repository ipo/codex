use std::cmp::Ordering;

use serde_json::Value;

const MAX_DECIMAL_SHIFT: u64 = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ExactJsonNumber {
    negative: bool,
    digits: String,
    exponent: i32,
}

impl ExactJsonNumber {
    pub(super) fn parse(number: &serde_json::Number) -> Result<Self, String> {
        let representation = number.to_string();
        let (mantissa, explicit_exponent) = representation
            .find(['e', 'E'])
            .map(|index| (&representation[..index], Some(&representation[index + 1..])))
            .unwrap_or((&representation, None));
        let explicit_exponent = explicit_exponent
            .map(|value| {
                value.parse::<i32>().map_err(|error| {
                    format!("invalid JSON number exponent in `{representation}`: {error}")
                })
            })
            .transpose()?
            .unwrap_or(0);
        let (negative, mantissa) = mantissa
            .strip_prefix('-')
            .map(|value| (true, value))
            .unwrap_or((false, mantissa));
        let fractional_digits = mantissa
            .split_once('.')
            .map(|(_, fractional)| fractional.len())
            .unwrap_or(0);
        let mut digits = mantissa
            .bytes()
            .filter(u8::is_ascii_digit)
            .map(char::from)
            .collect::<String>();
        let first_nonzero = digits
            .bytes()
            .position(|byte| byte != b'0')
            .unwrap_or(digits.len());
        digits.drain(..first_nonzero);
        if digits.is_empty() {
            return Ok(Self {
                negative: false,
                digits: "0".to_string(),
                exponent: 0,
            });
        }
        let mut exponent = explicit_exponent
            .checked_sub(i32::try_from(fractional_digits).map_err(|_| {
                format!("JSON number `{representation}` has too many fractional digits")
            })?)
            .ok_or_else(|| format!("JSON number `{representation}` exponent is out of range"))?;
        while digits.ends_with('0') {
            digits.pop();
            exponent = exponent.checked_add(1).ok_or_else(|| {
                format!("JSON number `{representation}` exponent is out of range")
            })?;
        }
        Ok(Self {
            negative,
            digits,
            exponent,
        })
    }

    pub(super) fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (negative, _) => {
                let magnitude = compare_magnitude(self, other);
                if negative {
                    magnitude.reverse()
                } else {
                    magnitude
                }
            }
        }
    }

    pub(super) fn is_positive(&self) -> bool {
        !self.negative && self.digits != "0"
    }

    pub(super) fn is_multiple_of(&self, divisor: &Self) -> Result<bool, String> {
        if self.digits == "0" {
            return Ok(true);
        }
        let exponent_difference = i64::from(self.exponent) - i64::from(divisor.exponent);
        let decimal_shift = exponent_difference.unsigned_abs();
        if decimal_shift > MAX_DECIMAL_SHIFT {
            return Err(format!(
                "exact `multipleOf` validation exceeds the {MAX_DECIMAL_SHIFT}-digit decimal shift limit"
            ));
        }
        let (mut numerator, mut denominator) = (self.digits.clone(), divisor.digits.clone());
        if exponent_difference >= 0 {
            numerator.extend(std::iter::repeat_n(
                '0',
                usize::try_from(decimal_shift).unwrap_or(usize::MAX),
            ));
        } else {
            denominator.extend(std::iter::repeat_n(
                '0',
                usize::try_from(decimal_shift).unwrap_or(usize::MAX),
            ));
        }
        Ok(decimal_integer_is_divisible(&numerator, &denominator))
    }
}

fn compare_magnitude(left: &ExactJsonNumber, right: &ExactJsonNumber) -> Ordering {
    let left_order =
        i64::try_from(left.digits.len()).unwrap_or(i64::MAX) + i64::from(left.exponent);
    let right_order =
        i64::try_from(right.digits.len()).unwrap_or(i64::MAX) + i64::from(right.exponent);
    left_order.cmp(&right_order).then_with(|| {
        let width = left.digits.len().max(right.digits.len());
        (0..width)
            .map(|index| {
                let left = left.digits.as_bytes().get(index).copied().unwrap_or(b'0');
                let right = right.digits.as_bytes().get(index).copied().unwrap_or(b'0');
                left.cmp(&right)
            })
            .find(|ordering| !ordering.is_eq())
            .unwrap_or(Ordering::Equal)
    })
}

fn decimal_integer_is_divisible(numerator: &str, denominator: &str) -> bool {
    let mut remainder = "0".to_string();
    for digit in numerator.bytes() {
        if remainder == "0" {
            remainder.clear();
        }
        remainder.push(char::from(digit));
        trim_leading_zeroes(&mut remainder);
        while compare_integer_digits(&remainder, denominator).is_ge() {
            remainder = subtract_integer_digits(&remainder, denominator);
        }
    }
    remainder == "0"
}

fn compare_integer_digits(left: &str, right: &str) -> Ordering {
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn subtract_integer_digits(left: &str, right: &str) -> String {
    let mut result = left.bytes().map(|byte| byte - b'0').collect::<Vec<_>>();
    let mut borrow = 0;
    for offset in 0..result.len() {
        let left_index = result.len() - 1 - offset;
        let right_digit = right
            .len()
            .checked_sub(offset + 1)
            .map(|index| right.as_bytes()[index] - b'0')
            .unwrap_or(0);
        let subtrahend = right_digit + borrow;
        if result[left_index] < subtrahend {
            result[left_index] += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result[left_index] -= subtrahend;
    }
    let mut result = result
        .into_iter()
        .map(|digit| char::from(digit + b'0'))
        .collect::<String>();
    trim_leading_zeroes(&mut result);
    result
}

fn trim_leading_zeroes(value: &mut String) {
    let first_nonzero = value
        .bytes()
        .position(|byte| byte != b'0')
        .unwrap_or(value.len().saturating_sub(1));
    value.drain(..first_nonzero);
}

pub(super) fn json_schema_equal(left: &Value, right: &Value) -> Result<bool, String> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => Ok(ExactJsonNumber::parse(left)?
            .cmp(&ExactJsonNumber::parse(right)?)
            .is_eq()),
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (left, right) in left.iter().zip(right) {
                if !json_schema_equal(left, right)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Value::Object(left), Value::Object(right)) => {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (key, left) in left {
                let Some(right) = right.get(key) else {
                    return Ok(false);
                };
                if !json_schema_equal(left, right)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(left == right),
    }
}

pub(super) fn contains_equivalent<'a>(
    values: impl IntoIterator<Item = &'a Value>,
    expected: &Value,
) -> Result<bool, String> {
    for value in values {
        if json_schema_equal(value, expected)? {
            return Ok(true);
        }
    }
    Ok(false)
}
