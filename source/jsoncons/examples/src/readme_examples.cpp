// Copyright 2013 Daniel Parker
// Distributed under Boost license

#include <jsoncons/json.hpp>
#include <jsoncons_ext/cbor/cbor.hpp>
#include <jsoncons_ext/jsonpointer/jsonpointer.hpp>
#include <jsoncons_ext/csv/csv_serializer.hpp>
#include <string>
#include <vector>
#include <iomanip>

namespace readme
{
    using namespace jsoncons;    

    void playing_around()
    {
        // Construct some CBOR using the streaming API
        std::vector<uint8_t> b;
        cbor::cbor_buffer_serializer writer(b);
        writer.begin_array(); // indefinite length outer array
        writer.begin_array(3); // a fixed length array
        writer.string_value("foo");
        writer.byte_string_value(byte_string{'P','u','s','s'}); // no suggested conversion
        writer.big_integer_value("-18446744073709551617");
        writer.end_array();
        writer.end_array();
        writer.flush();

        // Print bytes
        std::cout << "(1)\n";
        for (auto c : b)
        {
            std::cout << std::hex << std::setprecision(2) << std::setw(2)
                      << std::setfill('0') << static_cast<int>(c);
        }
        std::cout << "\n\n";
/*
        9f -- Start indefinte length array
          83 -- Array of length 3
            63 -- String value of length 3
              666f6f -- "foo" 
            44 -- Byte string value of length 4
              50757373 -- 'P''u''s''s'
            c3 -- Tag 3 (negative bignum)
              49 -- Byte string value of length 9
                010000000000000000 -- Bytes content
          ff -- "break" 
*/
        // Unpack bytes into a json variant value, and add some more elements
        json j = cbor::decode_cbor<json>(b);

        // Loop over the rows
        std::cout << "(2)\n";
        for (const json& row : j.array_range())
        {
            std::cout << row << "\n";
        }
        std::cout << "\n";

        // Get element at position 0/2 using jsonpointer 
        json& v = jsonpointer::get(j, "/0/2");
        std::cout << "(3) " << v.as<std::string>() << "\n\n";

        // Print JSON representation with default options
        std::cout << "(4)\n";
        std::cout << pretty_print(j) << "\n\n";

        // Print JSON representation with different options
        json_options options;
        options.byte_string_format(byte_string_chars_format::base64)
               .big_integer_format(big_integer_chars_format::base64url);
        std::cout << "(5)\n";
        std::cout << pretty_print(j, options) << "\n\n";

        // Add some more elements

        json another_array = json::array(); 
        another_array.emplace_back(byte_string({'P','u','s','s'}),
                                   semantic_tag_type::base64); // suggested conversion to base64
        another_array.emplace_back("273.15", semantic_tag_type::big_decimal);
        another_array.emplace(another_array.array_range().begin(),"bar"); // place at front

        j.push_back(std::move(another_array));
        std::cout << "(6)\n";
        std::cout << pretty_print(j) << "\n\n";

        // Get element at position /1/2 using jsonpointer
        json& ref = jsonpointer::get(j, "/1/2");
        std::cout << "(7) " << ref.as<std::string>() << "\n\n";

#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__STRICT_ANSI__) && defined(_GLIBCXX_USE_INT128))
        // e.g. if code compiled with GCC and std=gnu++11 (rather than std=c++11)
        __int128 i = j[1][2].as<__int128>();
#endif

        // Repack bytes
        std::vector<uint8_t> b2;
        cbor::encode_cbor(j, b2);

        // Print the repacked bytes
        std::cout << "(8)\n";
        for (auto c : b2)
        {
            std::cout << std::hex << std::setprecision(2) << std::setw(2)
                      << std::setfill('0') << static_cast<int>(c);
        }
        std::cout << "\n\n";
/*
        82 -- Array of length 2
          83 -- Array of length 3
            63 -- String value of length 3
              666f6f -- "foo" 
            44 -- Byte string value of length 4
              50757373 -- 'P''u''s''s'
            c3 -- Tag 3 (negative bignum)
            49 -- Byte string value of length 9
              010000000000000000 -- Bytes content
          83 -- Another array of length 3
          63 -- String value of length 3
            626172 -- "bar"
          d6 - Expected conversion to base64
          44 -- Byte string value of length 4
            50757373 -- 'P''u''s''s'
          c4 -- Tag 4 (decimal fraction)
            82 -- Array of length 2
              21 -- -2
              19 6ab3 -- 27315
*/
        // Serialize to CSV
        csv::csv_options csv_options;
        csv_options.column_names("Column 1,Column 2,Column 3");

        std::string csv;
        csv::encode_csv(j, csv, csv_options);
        std::cout << "(9)\n";
        std::cout << csv << "\n\n";
    }
}

void readme_examples()
{
    std::cout << "\nReadme examples\n\n";

    readme::playing_around();

    std::cout << std::endl;
}

