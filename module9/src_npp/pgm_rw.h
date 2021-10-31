#ifndef IGPUP_MODULE_9_PGM_RW_H
#define IGPUP_MODULE_9_PGM_RW_H

#include <cstdint>
#include <string>
#include <vector>

class PGMFile {
 public:
  PGMFile();

  /// Create PGMFile directly from data
  PGMFile(const std::uint8_t *const data, unsigned int size,
          unsigned int width);

  // allow copy
  PGMFile(const PGMFile &other) = default;
  PGMFile &operator=(const PGMFile &other) = default;

  // allow move
  PGMFile(PGMFile &&other) = default;
  PGMFile &operator=(PGMFile &&other) = default;

  bool LoadImage(const char *filename);
  bool LoadImage(const std::string &filename);

  bool SaveImage(const char *filename, bool overwrite = false);
  bool SaveImage(const std::string &filename, bool overwrite = false);

  bool IsLoaded() const;

  static bool FilenameEndsWithPGM(const std::string &filename);

  const std::vector<std::uint8_t> &GetImageData() const;

 private:
  std::vector<std::uint8_t> image_data_;
  unsigned int width;

  bool WidthHeightDataLocHelper(const std::string &filename,
                                unsigned int *width, unsigned int *height,
                                unsigned int *max_value,
                                unsigned int *data_idx);
  bool DecodeASCIIPGM(const std::string &filename);
  bool DecodeRAWPGM(const std::string &filename);
};

#endif
