#pragma once

#include "string.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace visp {

//
// JSON value - minimal, ordered tree
//
// A small self-contained implementation (no external dependency) for the
// HTTP server. Object keys keep insertion order and duplicate keys are
// allowed in parsed values (the last one wins on lookup).

struct json_value {
    enum class type { null, boolean, number, string, array, object };

    type t = type::null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<json_value> array;
    std::vector<std::pair<std::string, json_value>> object;

    json_value() = default;

    json_value(bool v) : t(type::boolean), boolean(v) {}
    json_value(int v) : t(type::number), number(v) {}
    json_value(int64_t v) : t(type::number), number(double(v)) {}
    json_value(size_t v) : t(type::number), number(double(v)) {}
    json_value(float v) : t(type::number), number(v) {}
    json_value(double v) : t(type::number), number(v) {}
    json_value(char const* v) : t(type::string), string(v) {}
    json_value(std::string const& v) : t(type::string), string(v) {}
    json_value(std::string&& v) : t(type::string), string(std::move(v)) {}
    json_value(std::string_view v) : t(type::string), string(v) {}

    bool is_null() const { return t == type::null; }
    bool is_bool() const { return t == type::boolean; }
    bool is_number() const { return t == type::number; }
    bool is_string() const { return t == type::string; }
    bool is_array() const { return t == type::array; }
    bool is_object() const { return t == type::object; }

    // Find a member of an object. Returns nullptr when absent or not an object.
    json_value* find(char const* key) {
        if (t != type::object) {
            return nullptr;
        }
        for (auto& [k, v] : object) {
            if (k == key) {
                return &v;
            }
        }
        return nullptr;
    }

    json_value const* find(char const* key) const {
        return const_cast<json_value*>(this)->find(key);
    }

    // Typed getters. Throw `except` (a user-recoverable error) when the
    // member is absent or has a different type, so callers can be written
    // without null checks.
    int get_int(char const* key) const {
        json_value const* v = find(key);
        if (!v) {
            throw except("Missing JSON member: {}", key);
        }
        if (!v->is_number()) {
            throw except("JSON member '{}' is not a number", key);
        }
        return int(v->number);
    }

    int64_t get_int64(char const* key) const {
        json_value const* v = find(key);
        if (!v) {
            throw except("Missing JSON member: {}", key);
        }
        if (!v->is_number()) {
            throw except("JSON member '{}' is not a number", key);
        }
        return int64_t(v->number);
    }

    double get_double(char const* key) const {
        json_value const* v = find(key);
        if (!v) {
            throw except("Missing JSON member: {}", key);
        }
        if (!v->is_number()) {
            throw except("JSON member '{}' is not a number", key);
        }
        return v->number;
    }

    bool get_bool(char const* key) const {
        json_value const* v = find(key);
        if (!v) {
            throw except("Missing JSON member: {}", key);
        }
        if (v->t != type::boolean) {
            throw except("JSON member '{}' is not a boolean", key);
        }
        return v->boolean;
    }

    std::string const& get_string(char const* key) const {
        json_value const* v = find(key);
        if (!v) {
            throw except("Missing JSON member: {}", key);
        }
        if (!v->is_string()) {
            throw except("JSON member '{}' is not a string", key);
        }
        return v->string;
    }

    // Add or replace a member of an object.
    json_value& set(char const* key, json_value&& value) {
        if (t != type::object) {
            *this = json_value();
            t = type::object;
        }
        if (json_value* v = find(key)) {
            *v = std::move(value);
            return *v;
        }
        object.emplace_back(key, std::move(value));
        return object.back().second;
    }
};

//
// Parser

namespace json_detail {

struct parser {
    std::string_view in;
    size_t pos = 0;
    int depth = 0;

    static constexpr int max_depth = 32;

    void skip_ws() {
        while (pos < in.size() &&
               (in[pos] == ' ' || in[pos] == '\t' || in[pos] == '\n' || in[pos] == '\r')) {
            ++pos;
        }
    }

    char peek() {
        if (pos >= in.size()) {
            throw except("Unexpected end of JSON input");
        }
        return in[pos];
    }

    void expect(char c) {
        if (peek() != c) {
            throw except("Expected '{}' in JSON, found '{}'", c, peek());
        }
        ++pos;
    }

    json_value parse() {
        skip_ws();
        json_value v = parse_value();
        skip_ws();
        if (pos != in.size()) {
            throw except("Trailing data after JSON value");
        }
        return v;
    }

    json_value parse_value() {
        if (++depth > max_depth) {
            throw except("JSON nesting too deep (max {})");
        }
        skip_ws();
        json_value v = [this]() {
            switch (peek()) {
                case 'n': expect_literal("null"); return json_value();
                case 't': expect_literal("true"); return json_value(true);
                case 'f': expect_literal("false"); return json_value(false);
                case '"': return json_value(parse_string());
                case '[': return parse_array();
                case '{': return parse_object();
                default: return parse_number();
            }
        }();
        --depth;
        return v;
    }

    void expect_literal(char const* lit) {
        size_t len = std::char_traits<char>::length(lit);
        if (in.substr(pos, len) != lit) {
            throw except("Invalid JSON literal");
        }
        pos += len;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        for (;;) {
            if (pos >= in.size()) {
                throw except("Unterminated JSON string");
            }
            char c = in[pos++];
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                if (pos >= in.size()) {
                    throw except("Unterminated JSON escape");
                }
                char e = in[pos++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        // A high surrogate must be followed by a low surrogate.
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (pos + 1 < in.size() && in[pos] == '\\' && in[pos + 1] == 'u') {
                                pos += 2;
                                uint32_t lo = parse_hex4();
                                if (lo < 0xDC00 || lo > 0xDFFF) {
                                    throw except("Invalid UTF-16 surrogate pair");
                                }
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                throw except("Unpaired UTF-16 surrogate");
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: throw except("Invalid JSON escape: \\{}", e);
                }
            } else if (uint8_t(c) < 0x20) {
                throw except("Unescaped control character in JSON string");
            } else {
                out += c;
            }
        }
        return out;
    }

    uint32_t parse_hex4() {
        if (pos + 4 > in.size()) {
            throw except("Truncated \\u escape");
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = in[pos++];
            v <<= 4;
            if (c >= '0' && c <= '9') {
                v |= uint32_t(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                v |= uint32_t(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                v |= uint32_t(c - 'A' + 10);
            } else {
                throw except("Invalid \\u escape: {}", c);
            }
        }
        return v;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    json_value parse_number() {
        size_t start = pos;
        if (peek() == '-') {
            ++pos;
        }
        while (pos < in.size() &&
               (std::isdigit(uint8_t(in[pos])) || in[pos] == '.' || in[pos] == 'e' ||
                in[pos] == 'E' || in[pos] == '+' || in[pos] == '-')) {
            ++pos;
        }
        if (start == pos) {
            throw except("Invalid JSON value at byte {}", pos);
        }
        std::string_view num = in.substr(start, pos - start);
        // Reject forms that strtod would silently accept but JSON does not
        // (eg. "01", "1e", "-"). A leading zero before a decimal point is
        // required for fractions below one ("0.5") and is valid.
        if (num.size() > 1 && num[0] == '0' && std::isdigit(uint8_t(num[1]))) {
            throw except("Invalid JSON number: {}", num);
        }
        char const* digits = num.data();
        char* end = nullptr;
        double v = std::strtod(digits, &end);
        if (end != digits + num.size() || !std::isfinite(v)) {
            throw except("Invalid JSON number: {}", num);
        }
        return json_value(v);
    }

    json_value parse_array() {
        expect('[');
        json_value v;
        v.t = json_value::type::array;
        skip_ws();
        if (peek() == ']') {
            ++pos;
            return v;
        }
        for (;;) {
            v.array.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') {
                ++pos;
            } else if (c == ']') {
                ++pos;
                return v;
            } else {
                throw except("Expected ',' or ']' in JSON array");
            }
        }
    }

    json_value parse_object() {
        expect('{');
        json_value v;
        v.t = json_value::type::object;
        skip_ws();
        if (peek() == '}') {
            ++pos;
            return v;
        }
        for (;;) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v.object.emplace_back(std::move(key), parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') {
                ++pos;
            } else if (c == '}') {
                ++pos;
                return v;
            } else {
                throw except("Expected ',' or '}' in JSON object");
            }
        }
    }
};

} // namespace json_detail

inline json_value json_parse(std::string_view text) {
    json_detail::parser p;
    p.in = text;
    return p.parse();
}

//
// Writer

namespace json_detail {

inline void escape_into(std::string& out, std::string_view s) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (uint8_t(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

inline void number_into(std::string& out, double v) {
    if (std::isfinite(v) && v == std::floor(v) && std::abs(v) <= 9007199254740992.0) {
        out += std::to_string(int64_t(v));
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        out += buf;
    }
}

inline void write_value(std::string& out, json_value const& v) {
    switch (v.t) {
        case json_value::type::null: out += "null"; break;
        case json_value::type::boolean: out += v.boolean ? "true" : "false"; break;
        case json_value::type::number: number_into(out, v.number); break;
        case json_value::type::string: escape_into(out, v.string); break;
        case json_value::type::array: {
            out += '[';
            for (size_t i = 0; i < v.array.size(); ++i) {
                if (i) {
                    out += ',';
                }
                write_value(out, v.array[i]);
            }
            out += ']';
            break;
        }
        case json_value::type::object: {
            out += '{';
            for (size_t i = 0; i < v.object.size(); ++i) {
                if (i) {
                    out += ',';
                }
                escape_into(out, v.object[i].first);
                out += ':';
                write_value(out, v.object[i].second);
            }
            out += '}';
            break;
        }
    }
}

} // namespace json_detail

inline void json_write(std::string& out, json_value const& v) {
    json_detail::write_value(out, v);
}

inline std::string json_write(json_value const& v) {
    std::string out;
    json_write(out, v);
    return out;
}

} // namespace visp
