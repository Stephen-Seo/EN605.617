#include "pgm_rw.h"

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>

PGMFile::PGMFile() : image_data_(), width(0) {}

PGMFile::PGMFile(const std::uint8_t *const data, unsigned int size,
                 unsigned int width)
    : image_data_(), width(width) {
  if (size % width != 0) {
    this->width = 0;
    return;
  }

  for (unsigned int i = 0; i < size; ++i) {
    image_data_.push_back(data[i]);
  }
}

bool PGMFile::LoadImage(const char *filename) {
  return LoadImage(std::string(filename));
}

bool PGMFile::LoadImage(const std::string &filename) {
  if (!FilenameEndsWithPGM(filename)) {
    return false;
  }

  std::ifstream ifs(filename);
  if (!(ifs.is_open() && ifs.good())) {
    std::cout << "ERROR: LoadImage failed, file doesn't exist or cannot be read"
              << std::endl;
    return false;
  }

  // get netpbm "magic number"
  std::string temp_str;
  try {
    ifs >> temp_str;
  } catch (const std::ios_base::failure &e) {
    std::cout << "ERROR: EOF or only whitespace in pgm file" << std::endl;
    return false;
  }
  if (temp_str.compare("P2") == 0) {
    // is ASCII (plain) pgm
    ifs.close();
    return DecodeASCIIPGM(filename);
  } else if (temp_str.compare("P5") == 0) {
    // is Binary (raw) pgm
    ifs.close();
    return DecodeRAWPGM(filename);
  } else {
    // invalid
    std::cout << "ERROR: Invalid \"magic number\" at start of pgm file"
              << std::endl;
    return false;
  }

  return IsLoaded();
}

bool PGMFile::SaveImage(const char *filename, bool overwrite) {
  if (!IsLoaded()) {
    return false;
  }
  return SaveImage(std::string(filename), overwrite);
}

bool PGMFile::SaveImage(const std::string &filename, bool overwrite) {
  if (!IsLoaded()) {
    return false;
  }

  if (image_data_.size() % this->width != 0) {
    // Somehow invalid data size, this shouldn't happen
    image_data_.clear();
    this->width = 0;
    return false;
  }

  if (!overwrite) {
    std::ifstream ifs(filename);
    if (ifs.is_open() && ifs.good()) {
      std::cout << "ERROR: Failed to SaveImage because file with same name "
                   "already exists and overwrite is set to false"
                << std::endl;
      return false;
    }
  }

  unsigned int height = image_data_.size() / width;

  std::ofstream ofs(filename);
  if (!(ofs.is_open() && ofs.good())) {
    std::cout << "ERROR: Failed to SaveImage, failed to open file for writing"
              << std::endl;
    return false;
  }

  ofs << "P5\n" << this->width << ' ' << height << "\n255\n";

  if (!(ofs.is_open() && ofs.good())) {
    std::cout << "ERROR: SaveImage: Something went wrong during writing header"
              << std::endl;
    return false;
  }

  ofs.close();
  ofs.open(filename,
           std::ios_base::out | std::ios_base::app | std::ios_base::binary);
  if (!(ofs.is_open() && ofs.good())) {
    std::cout << "ERROR: SaveImage: Failed to reopen file for binary writing"
              << std::endl;
    return false;
  }

  for (std::uint8_t value : image_data_) {
    ofs.put(value);
    if (!(ofs.is_open() && ofs.good())) {
      std::cout << "ERROR: SaveImage: Failure during binary writing"
                << std::endl;
      return false;
    }
  }

  return true;
}

bool PGMFile::IsLoaded() const { return !image_data_.empty() && width != 0; }

bool PGMFile::FilenameEndsWithPGM(const std::string &filename) {
  auto iter = filename.crbegin();
  if (*iter != 'm') {
    return false;
  }
  ++iter;
  if (*iter != 'g') {
    return false;
  }
  ++iter;
  if (*iter != 'p') {
    return false;
  }
  ++iter;
  if (*iter != '.') {
    return false;
  }

  return true;
}

const std::vector<std::uint8_t> &PGMFile::GetImageVector() const {
  return image_data_;
}

const std::uint8_t *PGMFile::GetImageData() const { return image_data_.data(); }

std::uint8_t *PGMFile::GetImageData() { return image_data_.data(); }

unsigned int PGMFile::GetSize() const { return image_data_.size(); }

unsigned int PGMFile::GetWidth() const { return width; }

unsigned int PGMFile::GetHeight() const {
  if (width != 0) {
    return image_data_.size() / width;
  }
  return 0;
}

bool PGMFile::WidthHeightDataLocHelper(const std::string &filename,
                                       unsigned int *width,
                                       unsigned int *height,
                                       unsigned int *max_value,
                                       unsigned int *data_idx) {
  std::ifstream ifs(filename);
  std::string temp;
  bool got_identifier = false;
  bool got_width = false;
  bool got_height = false;
  bool got_max_value = false;
  bool got_data_idx = false;
  bool got_comment = false;
  char next;

  /// returns false on error
  const auto do_checks_gets = [&got_identifier, &got_width, &got_height,
                               &got_max_value, width, height, max_value,
                               data_idx](const std::string &s) -> bool {
    if (!got_identifier) {
      if (s.compare("P2") == 0 || s.compare("P5") == 0) {
        got_identifier = true;
        return true;
      }
    } else if (!got_width) {
      try {
        *width = std::stoul(s);
      } catch (const std::invalid_argument &e) {
        std::cout << "ERROR: Failed to parse width" << std::endl;
        return false;
      } catch (const std::out_of_range &e) {
        std::cout << "ERROR: Failed to parse width" << std::endl;
        return false;
      }

      if (*width != 0) {
        got_width = true;
        return true;
      } else {
        std::cout << "ERROR: Got invalid width of \"0\"" << std::endl;
        return false;
      }
    } else if (!got_height) {
      try {
        *height = std::stoul(s);
      } catch (const std::invalid_argument &e) {
        std::cout << "ERROR: Failed to parse height" << std::endl;
        return false;
      } catch (const std::out_of_range &e) {
        std::cout << "ERROR: Failed to parse height" << std::endl;
        return false;
      }

      if (*height != 0) {
        got_height = true;
        return true;
      } else {
        std::cout << "ERROR: Got invalid height of \"0\"" << std::endl;
        return false;
      }
    } else if (!got_max_value) {
      try {
        *max_value = std::stoul(s);
      } catch (const std::invalid_argument &e) {
        std::cout << "ERROR: Failed to parse max_value" << std::endl;
        return false;
      } catch (const std::out_of_range &e) {
        std::cout << "ERROR: Failed to parse max_value" << std::endl;
        return false;
      }

      if (*max_value != 0) {
        got_max_value = true;
        return true;
      } else {
        std::cout << "ERROR: Got invalid max_value of \"0\"" << std::endl;
        return false;
      }
    }

    return false;
  };  // do_checks_get lambda fn

  while (ifs.is_open() && ifs.good()) {
    ifs.get(next);
    if (ifs.eof()) {
      break;
    }

    if (next == '#') {
      got_comment = true;
      continue;
    } else if (got_comment) {
      if (next == '\n') {
        got_comment = false;
        if (!temp.empty()) {
          do_checks_gets(temp);
          temp.clear();
        }
      }
      continue;
    } else if (next == ' ' || next == '\n') {
      if (!temp.empty()) {
        do_checks_gets(temp);
        temp.clear();
      }

      if (got_identifier && got_width && got_height && got_max_value) {
        auto idx = ifs.tellg();
        if (idx != -1) {
          *data_idx = static_cast<unsigned int>(idx);
          return true;
        } else {
          std::cout << "ERROR: Failed to get \"data_idx\"" << std::endl;
          return false;
        }
      }
    } else {
      temp.push_back(next);
    }
  }  // while (ifs.is_open() && ifs.good())

  std::cout << "End of WidthHeightDataLocHelper" << std::endl;
  return false;
}

bool PGMFile::DecodeASCIIPGM(const std::string &filename) {
  image_data_.clear();

  std::string temp;
  unsigned int temp_value = 0;
  unsigned int height = 0;
  unsigned int max_value = 0;
  unsigned int data_start_idx = 0;

  if (!WidthHeightDataLocHelper(filename, &this->width, &height, &max_value,
                                &data_start_idx)) {
    std::cout << "ERROR: Failed to get pertinent data from pgm file"
              << std::endl;
    return false;
  }

  std::ifstream ifs(filename);
  if (!(ifs.is_open() && ifs.good())) {
    std::cout << "ERROR: Failed to open file for parsing" << std::endl;
    return false;
  }

  ifs.seekg(data_start_idx);
  if (!(ifs.is_open() && ifs.good())) {
    std::cout << "ERROR: Failed to seek in file for parsing" << std::endl;
    return false;
  }

  while (ifs.is_open() && ifs.good()) {
    try {
      ifs >> temp;
    } catch (const std::ios_base::failure &e) {
      std::cout << "ERROR: EOF or only whitespace after index "
                << data_start_idx << std::endl;
      return false;
    }
    if (ifs.eof()) {
      break;
    }

    try {
      temp_value = std::stoul(temp);
    } catch (const std::invalid_argument &e) {
      std::cout << "ERROR: Failed to parse image data value" << std::endl;
      return false;
    } catch (const std::out_of_range &e) {
      std::cout << "ERROR: Failed to parse image data value" << std::endl;
      return false;
    }

    image_data_.push_back(static_cast<std::uint8_t>((float)temp_value /
                                                    (float)max_value * 255.0F));
  }

  if (image_data_.size() == this->width * height) {
    return true;
  } else {
    std::cout << "ERROR: Input data is invalid size: Got size == "
              << image_data_.size() << ", should be == " << this->width * height
              << std::endl;
    image_data_.clear();
    this->width = 0;
    return false;
  }
}

bool PGMFile::DecodeRAWPGM(const std::string &filename) {
  image_data_.clear();

  std::string temp;
  // temp_char_buf is used for possible 16-bit entries, so alignment is needed
  alignas(32) std::array<char, 3> temp_char_buf;
  std::uint16_t temp_16bit = 0;
  unsigned int height = 0;
  unsigned int max_value = 0;
  unsigned int data_start_idx = 0;
  bool is_8_bit = true;

  if (!WidthHeightDataLocHelper(filename, &this->width, &height, &max_value,
                                &data_start_idx)) {
    std::cout << "ERROR: Failed to get pertinent data from pgm file"
              << std::endl;
    return false;
  }

  if (max_value != 255 && max_value != 65535) {
    // P5 pgm must be 8-bit or 16-bit
    std::cout << "ERROR: RAW pgm has invalid max_value (must be 8-bit or "
                 "16-bit)"
              << std::endl;
    return false;
  } else if (max_value == 255) {
    is_8_bit = true;
  } else /* if (max_value == 65535) */ {
    is_8_bit = false;
  }

  std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);
  if (!(ifs.is_open() && ifs.good())) {
    std::cout << "ERROR: Failed to open file for RAW parsing" << std::endl;
    return false;
  }

  ifs.seekg(data_start_idx);
  if (!(ifs.is_open() && ifs.good())) {
    std::cout << "ERROR: Failed to seek in file for RAW parsing" << std::endl;
    return false;
  }

  while (ifs.is_open() && ifs.good()) {
    if (is_8_bit) {
      ifs.get(temp_char_buf[0]);
      if (ifs.eof()) {
        break;
      }
      image_data_.push_back(temp_char_buf[0]);
    } else {
      // 16 bit reads may need more testing due to possible endianness
      // differences
      ifs.get(temp_char_buf.data(), 3);  // ifs.get reads at most "count - 1"
      if (ifs.eof()) {
        break;
      }
      temp_16bit = *reinterpret_cast<std::uint16_t *>(temp_char_buf.data());
      image_data_.push_back((float)temp_16bit / 65535.0F * 255.0F);
    }
  }

  if (image_data_.size() == this->width * height) {
    return true;
  } else {
    std::cout << "ERROR: Input data is invalid size: Got size == "
              << image_data_.size() << ", should be == " << this->width * height
              << std::endl;
    image_data_.clear();
    this->width = 0;
    return false;
  }
}
