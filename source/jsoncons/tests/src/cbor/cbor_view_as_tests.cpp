// Copyright 2016 Daniel Parker
// Distributed under Boost license

#if defined(_MSC_VER)
#include "windows.h"
#endif
#include <jsoncons/json.hpp>
#include <jsoncons_ext/cbor/cbor.hpp>
#include <jsoncons_ext/jsonpointer/jsonpointer.hpp>
#include <sstream>
#include <vector>
#include <utility>
#include <ctime>
#include <limits>
#include <catch/catch.hpp>

using namespace jsoncons;
using namespace jsoncons::cbor;

#if !defined(JSONCONS_NO_DEPRECATED)

TEST_CASE("cbor_view array as<> test")
{
    std::vector<uint8_t> bytes;
    cbor::cbor_buffer_serializer writer(bytes);
    writer.begin_array(); // indefinite length outer array
    writer.string_value("foo");
    writer.byte_string_value(byte_string({'b','a','r'}));
    writer.big_integer_value("-18446744073709551617");
    writer.big_decimal_value("273.15");
    writer.date_time_value("2015-05-07 12:41:07-07:00");
    writer.timestamp_value(1431027667);
    writer.int64_value(-1431027667, semantic_tag_type::timestamp);
    writer.double_value(1431027667.5, semantic_tag_type::timestamp);
    writer.end_array();
    writer.flush();

/*
9f -- Start indefinite length array 
  63 -- String value of length 3 
    666f6f -- "foo"
  43 -- Byte string value of length 3
    626172 -- 'b''a''r'
  c3 -- Tag 3 (negative bignum)
    49 Byte string value of length 9
      010000000000000000 -- Bytes content
  c4  - Tag 4 (decimal fraction)
    82 -- Array of length 2
      21 -- -2
      19 6ab3 -- 27315
  c0 -- Tag 0 (date-time)
    78 19 -- Length (25)
      323031352d30352d30372031323a34313a30372d30373a3030 -- "2015-05-07 12:41:07-07:00"
  c1 -- Tag 1 (epoch time)
    1a -- uint32_t
      554bbfd3 -- 1431027667 
  c1
    3a
      554bbfd2
  c1
    fb
      41d552eff4e00000
  ff -- "break" 
*/

    //std::cout << "bytes: \n";
    //for (auto c : bytes)
    //{
    //    std::cout << std::hex << std::setprecision(2) << std::setw(2)
    //              << std::setfill('0') << static_cast<int>(c);
    //}
    //std::cout << "\n\n";

    cbor::cbor_view v = bytes; // a non-owning view of the CBOR bytes

    CHECK(v.size() == 8);

    SECTION("v[0].is<T>()")
    {
        CHECK(v[0].is<std::string>());
        CHECK(v[1].is<byte_string>());
        CHECK(v[1].is<byte_string_view>());
        CHECK(v[2].is<byte_string>());
        CHECK(v[2].is<byte_string_view>());
        CHECK(v[3].is_array());
        CHECK(v[4].is<std::string>());
        CHECK(v[5].is<int>());
        CHECK(v[5].is<unsigned int>());
        CHECK(v[6].is<int>());
        CHECK_FALSE(v[6].is<unsigned int>());
        CHECK(v[7].is<double>());
    }

    SECTION("v[0].as<T>()")
    {
        CHECK(v[0].as<std::string>() == std::string("foo"));
        CHECK(v[1].as<jsoncons::byte_string>() == jsoncons::byte_string({'b','a','r'}));
        CHECK(v[2].as<std::string>() == std::string("-18446744073709551617"));
        CHECK(bool(v[2].as<jsoncons::bignum>() == jsoncons::bignum("-18446744073709551617")));
        CHECK(v[3].as<std::string>() == std::string("273.15"));
        CHECK(v[4].as<std::string>() == std::string("2015-05-07 12:41:07-07:00"));
        CHECK(v[5].as<int64_t>() == 1431027667);
        CHECK(v[5].as<uint64_t>() == 1431027667U);
        CHECK(v[6].as<int64_t>() == -1431027667);
        CHECK(v[7].as<double>() == 1431027667.5);
    }

    SECTION("array_iterator is<T> test")
    {
        auto it = v.array_range().begin();
        CHECK(it++->is<std::string>());
        CHECK(it++->is<byte_string>());
        CHECK(it++->is<byte_string>());
        CHECK(it++->is_array());
        CHECK(it++->is<std::string>());
        CHECK(it++->is<int>());
        CHECK(it++->is<int>());
        CHECK(it++->is<double>());
    }
}
#endif

