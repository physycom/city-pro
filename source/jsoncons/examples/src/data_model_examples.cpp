// Copyright 2017 Daniel Parker
// Distributed under Boost license

#include <jsoncons/json.hpp>
#include <jsoncons_ext/cbor/cbor.hpp>
#include <iomanip>

using namespace jsoncons;

void data_model_example1()
{
    json j = json::array();

    j.emplace_back("foo");
    j.emplace_back(byte_string{ 'b','a','r' });
    j.emplace_back("-18446744073709551617", semantic_tag_type::big_integer);
    j.emplace_back("273.15", semantic_tag_type::big_decimal);
    j.emplace_back("2018-10-19 12:41:07-07:00", semantic_tag_type::date_time);
    j.emplace_back(1431027667, semantic_tag_type::timestamp);
    j.emplace_back(-1431027667, semantic_tag_type::timestamp);
    j.emplace_back(1431027667.5, semantic_tag_type::timestamp);

    std::cout << "(1)\n" << pretty_print(j) << "\n\n";

    std::vector<uint8_t> bytes;
    cbor::encode_cbor(j, bytes);
    std::cout << "(2)\n";
    for (auto c : bytes)
    {
        std::cout << std::hex << std::noshowbase << std::setprecision(2) << std::setw(2)
                  << std::setfill('0') << static_cast<int>(c);
    }
    std::cout << "\n\n";
/*
88 -- Array of length 8
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
      323031382d31302d31392031323a34313a30372d30373a3030 -- "2018-10-19 12:41:07-07:00"
  c1 -- Tag 1 (epoch time)
    1a -- uint32_t
      554bbfd3 -- 1431027667 
  c1
    3a
      554bbfd2
  c1
    fb
      41d552eff4e00000
*/
}

void data_model_example2()
{
    std::vector<uint8_t> bytes;
    cbor::cbor_buffer_serializer writer(bytes);
    writer.begin_array(); // indefinite length outer array
    writer.string_value("foo");
    writer.byte_string_value(byte_string{'b','a','r'});
    writer.big_integer_value("-18446744073709551617");
    writer.big_decimal_value("273.15");
    writer.date_time_value("2018-10-19 12:41:07-07:00");
    writer.timestamp_value(1431027667);
    writer.int64_value(-1431027667, semantic_tag_type::timestamp);
    writer.double_value(1431027667.5, semantic_tag_type::timestamp);
    writer.end_array();
    writer.flush();

    std::cout << "(1)\n";
    for (auto c : bytes)
    {
        std::cout << std::hex << std::setprecision(2) << std::setw(2) 
                  << std::noshowbase << std::setfill('0') << static_cast<int>(c);
    }
    std::cout << "\n\n";

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
      323031382d31302d31392031323a34313a30372d30373a3030 -- "2018-10-19 12:41:07-07:00"
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

    json j = cbor::decode_cbor<json>(bytes);

    std::cout << "(2)\n" << pretty_print(j) << "\n\n";
}

void data_model_examples()
{
    std::cout << "\ndata model examples\n\n";
    data_model_example1();
    data_model_example2();
    std::cout << std::endl;
}

