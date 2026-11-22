#include "testing.h"
#include "util/json.h"

namespace visp {

VISP_TEST(json_null) {
    json_value v = json_parse("null");
    CHECK(v.is_null());
}

VISP_TEST(json_bool) {
    json_value v = json_parse("{\"b\": true}");
    CHECK(v.get_bool("b"));
}

VISP_TEST(json_number) {
    json_value v = json_parse("-12.5e+2");
    CHECK(v.is_number());
    CHECK_EQUAL(v.number, -1250.0);
}

VISP_TEST(json_number_rejects_leading_zero) {
    bool failed = false;
    try {
        json_parse("012");
    } catch (std::exception const&) {
        failed = true;
    }
    CHECK(failed);
}

VISP_TEST(json_string_escapes) {
    json_value v = json_parse("\"a\\\"b\\\\c\\n\\ud83d\\ude00\"");
    CHECK(v.is_string());
    CHECK_EQUAL(v.string, std::string("a\"b\\c\n") + "\xF0\x9F\x98\x80");
}

VISP_TEST(json_string_rejects_control) {
    bool failed = false;
    try {
        json_parse("\"a\nb\"");
    } catch (std::exception const&) {
        failed = true;
    }
    CHECK(failed);
}

VISP_TEST(json_array) {
    json_value v = json_parse("[1, 2.5, \"three\", [4], {}]");
    CHECK(v.is_array());
    CHECK_EQUAL(v.array.size(), size_t(5));
    CHECK(v.array[3].is_array());
    CHECK(v.array[4].is_object());
}

VISP_TEST(json_object) {
    json_value v = json_parse("{ \"b\": 2, \"a\": 1 }");
    CHECK(v.is_object());
    CHECK_EQUAL(v.object.size(), size_t(2));
    // insertion order is preserved
    CHECK_EQUAL(v.object[0].first, std::string("b"));
    CHECK_EQUAL(v.find("a")->number, 1.0);
    CHECK(v.find("missing") == nullptr);
}

VISP_TEST(json_nested) {
    json_value v = json_parse("{\"a\": {\"b\": [{\"c\": true}]}}");
    json_value const* c = v.find("a")->find("b")->array[0].find("c");
    CHECK(c != nullptr);
    CHECK(c->is_bool() && c->boolean);
}

VISP_TEST(json_getters) {
    json_value v = json_parse("{\"i\": 42, \"d\": 1.5, \"s\": \"x\", \"b\": false}");
    CHECK_EQUAL(v.get_int("i"), 42);
    CHECK_EQUAL(v.get_double("d"), 1.5);
    CHECK_EQUAL(v.get_string("s"), std::string("x"));
    CHECK(v.get_bool("b") == false);
}

VISP_TEST(json_getter_missing) {
    json_value v = json_parse("{}");
    bool failed = false;
    try {
        v.get_int("nope");
    } catch (std::exception const&) {
        failed = true;
    }
    CHECK(failed);
}

VISP_TEST(json_getter_wrong_type) {
    json_value v = json_parse("{\"n\": \"not a number\"}");
    bool failed = false;
    try {
        v.get_int("n");
    } catch (std::exception const&) {
        failed = true;
    }
    CHECK(failed);
}

VISP_TEST(json_set) {
    json_value v;
    v.set("a", 1);
    v.set("b", std::string("two"));
    v.set("a", 3); // replace
    CHECK_EQUAL(v.object.size(), size_t(2));
    CHECK_EQUAL(v.find("a")->number, 3.0);
}

VISP_TEST(json_write_roundtrip) {
    json_value v;
    v.set("name", std::string("he said \"hi\" \n"));
    v.set("count", 3);
    json_value arr;
    arr.t = json_value::type::array;
    arr.array.push_back(1.5);
    arr.array.push_back(json_value()); // null
    v.set("items", std::move(arr));

    std::string text = json_write(v);
    json_value r = json_parse(text);
    CHECK_EQUAL(r.get_string("name"), std::string("he said \"hi\" \n"));
    CHECK_EQUAL(r.get_int("count"), 3);
    CHECK_EQUAL(r.find("items")->array.size(), size_t(2));
    CHECK(r.find("items")->array[1].is_null());
}

VISP_TEST(json_write_integers_stay_integers) {
    json_value v;
    v.set("a", 42);
    v.set("b", 0.5);
    std::string text = json_write(v);
    CHECK(text.find("42,") != std::string::npos);
    CHECK(text.find("42.0") == std::string::npos);
}

VISP_TEST(json_trailing_data_rejected) {
    bool failed = false;
    try {
        json_parse("1 2");
    } catch (std::exception const&) {
        failed = true;
    }
    CHECK(failed);
}

VISP_TEST(json_depth_limit) {
    std::string deep(100, '[');
    bool failed = false;
    try {
        json_parse(deep);
    } catch (std::exception const&) {
        failed = true;
    }
    CHECK(failed);
}

VISP_TEST(json_base64_roundtrip) {
    std::vector<std::byte> data{
        std::byte(0), std::byte(1), std::byte(2), std::byte(255), std::byte(128), std::byte(64)};
    std::string enc = base64_encode(data);
    auto dec = base64_decode(enc);
    CHECK_EQUAL(dec.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        CHECK_EQUAL(uint8_t(dec[i]), uint8_t(data[i]));
    }
    // single and double remainder
    CHECK(
        base64_decode(base64_encode(std::vector<std::byte>{std::byte(7)})) ==
        std::vector<std::byte>{std::byte(7)});
    CHECK(
        base64_decode(base64_encode(std::vector<std::byte>{std::byte(1), std::byte(2)})) ==
        std::vector<std::byte>{std::byte(1), std::byte(2)});
}

VISP_TEST(json_base64_rejects_garbage) {
    bool failed = false;
    try {
        base64_decode("!!!!");
    } catch (std::exception const&) {
        failed = true;
    }
    CHECK(failed);
}

} // namespace visp
