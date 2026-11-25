function(mgcm_add_slang_shader OUT_HEADERS_VAR SLANG_REL_PATH ENTRY_NAME VAR_NAME)
    set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${SLANG_REL_PATH}")
    set(SPV_FILE "${CMAKE_CURRENT_BINARY_DIR}/spv/${VAR_NAME}.spv")
    set(HDR_FILE "${CMAKE_CURRENT_BINARY_DIR}/generated_spirv/${VAR_NAME}.hpp")

    if(MGCM_COMPILE_SLANG)
        set(VAL_CMD "")
        if(SPIRV_VAL_EXECUTABLE)
            set(VAL_CMD COMMAND "${SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${SPV_FILE}")
        endif()

        add_custom_command(
            OUTPUT "${HDR_FILE}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/spv"
            COMMAND "${SLANGC_EXECUTABLE}" "${SRC_FILE}"
                    -target spirv -profile spirv_1_6
                    -emit-spirv-directly -fvk-use-entrypoint-name
                    -entry "${ENTRY_NAME}" -o "${SPV_FILE}"
            ${VAL_CMD}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py"
                    "${SPV_FILE}" "${HDR_FILE}" "${VAR_NAME}"
            DEPENDS "${SRC_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spv.py"
            COMMENT "[Slang -> SPIR-V 1.6 (VK 1.3)] ${ENTRY_NAME} -> ${VAR_NAME}.hpp"
            VERBATIM
        )
    endif()

    set(TMP_LIST ${${OUT_HEADERS_VAR}})
    list(APPEND TMP_LIST "${HDR_FILE}")
    set(${OUT_HEADERS_VAR} "${TMP_LIST}" PARENT_SCOPE)
endfunction()
