# ---------------------------------------------------------------------------
# Vendored from instantX-research/InstantID (https://github.com/instantX-research/InstantID)
# Licensed under the Apache License, Version 2.0 (see LICENSE-InstantID below).
# Fetched 2026-07-25 from the "main" branch. Treat as frozen vendor code: do not
# edit except for compatibility patches, and note any patch in a comment here.
#
# Copyright 2024 The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
import torch.nn.functional as F


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
