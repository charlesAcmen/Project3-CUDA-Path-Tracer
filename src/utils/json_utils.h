#pragma once

// Helpers for the project's hand-authored JSON files. Object keys are ASCII
// case-insensitive; string values such as asset paths and material names keep
// their original, case-sensitive semantics.

#include <json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace JsonUtil {

inline bool asciiCaseInsensitiveEqual(std::string_view left, std::string_view right)
{
    if (left.size() != right.size()) return false;

    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const auto fold = [](unsigned char c) {
            return c >= 'A' && c <= 'Z' ? static_cast<unsigned char>(c + ('a' - 'A')) : c;
        };
        if (fold(static_cast<unsigned char>(left[i])) !=
            fold(static_cast<unsigned char>(right[i])))
        {
            return false;
        }
    }
    return true;
}

// An exact spelling wins. Otherwise, one differently cased spelling is
// accepted. Multiple non-exact matches are rejected rather than picked
// arbitrarily.
inline const nlohmann::json* findKey(const nlohmann::json& object, std::string_view key)
{
    if (!object.is_object()) return nullptr;

    const auto exact = object.find(std::string(key));
    if (exact != object.end()) return &exact.value();

    const nlohmann::json* match = nullptr;
    for (auto it = object.begin(); it != object.end(); ++it)
    {
        if (!asciiCaseInsensitiveEqual(it.key(), key)) continue;
        if (match != nullptr)
        {
            throw std::runtime_error("Ambiguous JSON keys after case-insensitive lookup: '" +
                                     std::string(key) + "'");
        }
        match = &it.value();
    }
    return match;
}

inline const nlohmann::json& requireKey(const nlohmann::json& object, std::string_view key)
{
    if (const auto* value = findKey(object, key)) return *value;
    throw std::out_of_range("Missing required JSON key: '" + std::string(key) + "'");
}

template <typename T>
T valueOr(const nlohmann::json& object, std::string_view key, T fallback)
{
    if (const auto* value = findKey(object, key)) return value->get<T>();
    return fallback;
}

} // namespace JsonUtil
