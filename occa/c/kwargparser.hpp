/* The MIT License (MIT)
 *
 * Copyright (c) 2018 David Medina
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_PY_KWARGPARSER_HEADER
#define OCCA_PY_KWARGPARSER_HEADER

#include <map>

#include "fromPy.hpp"
#include "types.hpp"

namespace occa {
  namespace py {
    namespace argType {
      enum type {
        long_long,
        string,
        dim,
        ptr,
        list,
        device,
        memory,
        kernel,
        stream,
        streamTag,
        properties,
      };
    }

    class kwargParser {
    public:
      std::string format;

      std::vector<argType::type> argTypes;
      std::vector<std::string> kwargNames;
      std::vector<void*> inputs;

      inline kwargParser() {}

      inline kwargParser& startOptionalKwargs() {
        format += '|';
        return *this;
      }

      //---[ Input ]--------------------
#define DEFINE_ADD(INPUT_TYPE, ARG_TYPE, FORMAT)            \
      inline kwargParser& add(const std::string &kwargName, \
                              INPUT_TYPE &input) {          \
        format += FORMAT;                                   \
        argTypes.push_back(ARG_TYPE);                       \
        kwargNames.push_back(kwargName);                    \
        inputs.push_back((void*) &input);                   \
        return *this;                                       \
      }

      DEFINE_ADD(std::string     , argType::string    , 's')
      DEFINE_ADD(long long       , argType::long_long , 'L')
      DEFINE_ADD(void*           , argType::ptr       , 'O')
      DEFINE_ADD(occa::py::list  , argType::list      , 'O')
      DEFINE_ADD(occa::dim       , argType::dim       , 'O')
      DEFINE_ADD(occa::device    , argType::device    , 'O')
      DEFINE_ADD(occa::memory    , argType::memory    , 'O')
      DEFINE_ADD(occa::kernel    , argType::kernel    , 'O')
      DEFINE_ADD(occa::stream    , argType::stream    , 'O')
      DEFINE_ADD(occa::streamTag , argType::streamTag , 'O')
      DEFINE_ADD(occa::properties, argType::properties, 's')

#undef DEFINE_ADD
      //================================

      //---[ Input Transforms ]---------
      template <class InputType, class ValueType>
      inline void setInput(const int index,
                           void *value) {
        *((InputType*) inputs[index]) = (ValueType) value;
      }

      template <class InputType, class ValueType>
      inline void setPtrInput(const int index,
                              void *value) {
        setInput<InputType, ValueType>(index,
                                       occa::py::ptr((PyObject*) value));
      }
      //================================

      //---[ Arg Setters ]--------------
      inline void setLongLong(const int index,
                              void *value) {
        ::memcpy(inputs[index], &value, sizeof(long long));
      }

      inline void setString(const int index,
                            void *value) {
        setInput<std::string, char*>(index, value);
      }

      inline void setPtr(const int index,
                         void *value) {
        setPtrInput<void*, void*>(index, value);
      }

      inline void setList(const int index,
                          void *value) {
        occa::py::list &input = *((occa::py::list*) inputs[index]);
        input.setObj((PyObject*) value);
      }

      inline void setDim(const int index,
                         void *value) {

        occa::py::list list((PyObject*) value);
        const int listSize = list.size();

        occa::dim &input = *((occa::dim*) inputs[index]);
        if (listSize >= 0) {
          input.x = occa::py::longlong(list[0]);
          if (listSize >= 1) {
            input.y = occa::py::longlong(list[1]);
            if (listSize >= 2) {
              input.z = occa::py::longlong(list[2]);
            }
          }
        }
      }

      inline void setDevice(const int index,
                            void *value) {
        setPtrInput<occa::device, occa::modeDevice_t*>(index, value);
      }

      inline void setMemory(const int index,
                            void *value) {
        setPtrInput<occa::memory, occa::modeMemory_t*>(index, value);
      }

      inline void setKernel(const int index,
                            void *value) {
        setPtrInput<occa::kernel, occa::modeKernel_t*>(index, value);
      }

      inline void setStream(const int index,
                            void *value) {
        setPtrInput<occa::stream, occa::modeStream_t*>(index, value);
      }

      inline void setStreamTag(const int index,
                               void *value) {
        setPtrInput<occa::streamTag, occa::modeStreamTag_t*>(index, value);
      }

      inline void setProperties(const int index,
                                void *value) {
        *((occa::properties*) inputs[index]) = occa::properties((char*) value);
      }

      inline void setArgValue(const int index,
                              void *value) {
        if (!value) {
          return;
        }
        switch (argTypes[index]) {
        case argType::long_long : return setLongLong(index, value);
        case argType::string    : return setString(index, value);
        case argType::ptr       : return setPtr(index, value);
        case argType::list      : return setList(index, value);
        case argType::dim       : return setDim(index, value);
        case argType::device    : return setDevice(index, value);
        case argType::memory    : return setMemory(index, value);
        case argType::kernel    : return setKernel(index, value);
        case argType::stream    : return setStream(index, value);
        case argType::streamTag : return setStreamTag(index, value);
        case argType::properties: return setProperties(index, value);
        }
      }
      //================================

      inline bool parse(PyObject *args, PyObject *kwargs) {
        OCCA_TRY_AND_RETURN(
          false,
          return unsafeParse(args, kwargs);
        );
      }

      inline bool unsafeParse(PyObject *args, PyObject *kwargs) {
        const int argCount = (int) argTypes.size();
        bool success = false;

        std::vector<const char*> kwargNamesPtrs;
        for (int i = 0; i < argCount; ++i) {
          kwargNamesPtrs.push_back(kwargNames[i].c_str());
        }
        kwargNamesPtrs.push_back(NULL);

#define PARSE_FOR(NUM, ...)                                             \
        case NUM: success = (                                           \
          PyArg_ParseTupleAndKeywords(args, kwargs,                     \
                                      format.c_str(),                   \
                                      (char**) &(kwargNamesPtrs[0]),    \
                                      __VA_ARGS__)                      \
        ); break

        void *a[9] = {
          NULL, NULL, NULL,
          NULL, NULL, NULL,
          NULL, NULL, NULL,
        };
        switch (argCount) {
          PARSE_FOR(1, &a[0]);
          PARSE_FOR(2, &a[0], &a[1]);
          PARSE_FOR(3, &a[0], &a[1], &a[2]);
          PARSE_FOR(4, &a[0], &a[1], &a[2], &a[3]);
          PARSE_FOR(5, &a[0], &a[1], &a[2], &a[3], &a[4]);
          PARSE_FOR(6, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5]);
          PARSE_FOR(7, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6]);
          PARSE_FOR(8, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
          PARSE_FOR(9, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7], &a[8]);
        }
#undef PARSE_FOR
        if (!success) {
          return false;
        }

        for (int i = 0; i < argCount; ++i) {
          setArgValue(i, a[i]);
        }
        return true;
      }
    };
  }
}

#endif
