#ifndef PARSE_CMD_LINE_HPP
#define PARSE_CMD_LINE_HPP

#include <string>
#include <sstream>
#include <iostream>

//////////////////////////////////////////////////////////////////////////////
// helper methods
//////////////////////////////////////////////////////////////////////////////

bool check_alpha(const std::string& option) {

    if(option.size() == 0) {
        std::cout << "ERROR: empty string as option not allowed" << std::endl;
        return false;
    }

    return true;
}

bool check_number(const std::string& option) {

    if(!check_alpha(option))
        return false;

    size_t value; std::stringstream sstrm; sstrm << option; sstrm >> value;
    if (std::to_string(value).compare(option)) {
        std::cout << "ERROR: option is not an unsigned integer: "
                  << option <<  std::endl;
        return false;
    }

    return true;
}

size_t to_number(const std::string& option) {
    size_t value; std::stringstream sstrm; sstrm << option; sstrm >> value;
    return value;
}

// borrowed from stack overflow
// http://stackoverflow.com/questions/865668/parse-command-line-arguments
std::string get_cmd_option(char ** begin, char ** end,
                           const std::string& option) {

    char ** itr = find(begin, end, option);
    if (itr != end && ++itr != end) {
        return std::string(*itr);
    }

    return "";
}

// borrowed from stack overflow
// http://stackoverflow.com/questions/865668/parse-command-line-arguments
bool cmd_option_exists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

// FUSE really needs absolute paths
std::string real_path(const std::string& relative_path) {
    char resolved_path[PATH_MAX];
    if (!realpath(relative_path.c_str(), resolved_path))
        return std::string("");
    return std::string(resolved_path);
}


#endif
