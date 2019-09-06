-- Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--    http://www.apache.org/licenses/LICENSE-2.0
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- ============================================================================
local make_map = require 'common.make_map'
local custom_decals = require 'decorators.custom_decals_decoration'
local custom_entities = require 'common.custom_entities'
local custom_observations = require 'decorators.custom_observations'
local datasets_selector = require 'datasets.selector'
local debug_observations = require 'decorators.debug_observations'
local game = require 'dmlab.system.game'
local image_utils = require 'image_utils'
local map_maker = require 'dmlab.system.map_maker'
local maze_generation = require 'dmlab.system.maze_generation'
local pickup_decorator = require 'decorators.human_recognisable_pickups'
local random = require 'common.random'
local setting_overrides = require 'decorators.setting_overrides'
local texture_sets = require 'themes.texture_sets'
local themes = require 'themes.themes'
local hrp = require 'common.human_recognisable_pickups'

local DEFAULTS = {
    EXPLORE_MAP_MODE = 'PASSIVE',
    EPISODE_LENGTH_SECONDS = 30,
    SECOND_ORDER_EXPLORE_LENGTH_SECONDS = 4,
    EXPLORE_LENGTH_SECONDS = 10,
    DISTRACTOR_LENGTH_SECONDS = 10,
    PRE_EXPLORE_DISTRACTOR_LENGTH_SECONDS = 0,
    NUM_IMAGES = 4,
    CORRECT_REWARD = 10,
    INCORRECT_REWARD = 1,
    IMAGE_SCALE = 3.0,
    IMAGE_ROOM_HEIGHT = 4,
    SHOW_KEY_COLOR_SQUARE_SECONDS = 1,
    DISTRACTOR_ROOM_SIZE = {11, 11},
    SECOND_ORDER_EXPLORE_ROOM_SIZE = {3, 3},
    PROB_APPLE_IN_DISTRACTOR_MAP = 0.3,
    APPLE_REWARD = 5,
    APPLE_REWARD_PROB = 1.0,
    APPLE_EXTRA_REWARD_RANGE = 0,
    DIFFERENT_DISTRACT_ROOM_TEXTURE = false,
    DIFFERENT_REWARD_ROOM_TEXTURE = false,
    DIFFERENT_SECOND_ORDER_ROOM_TEXTURE = false,
}

local APPLE_ID = 999
local KEY_OBJECT_SPAWN_ID = 1000
local DOOR_ID = 1001

-- Table that maps from full decal name to decal index number.
local decalIndices = {}

local SECOND_ORDER_EXPLORE_MAP = "secondOrderExploreMap"
local EXPLORE_MAP = "exploreMap"
local DISTRACTOR_MAP = "distractorMap"
local IMAGE_MAP = "imageMap"
local COLORS = image_utils.COLORS

-- Set texture set for all maps.
local textureSet = texture_sets.PACMAN
local secondTextureSet = texture_sets.TETRIS
local thirdTextureSet = texture_sets.TRON
local fourthTextureSet = texture_sets.MINESWEEPER

local SHORT_STRAIGHT_ROOM =[[
***
*P*
* *
* *
***
]]

local SHORT_STRAIGHT_ROOM_WITH_DOOR =[[
***
*P*
*H*
* *
***
]]

local TWO_ROOMS = [[
*********
*********
*   *   *
* P     *
*   *   *
*********
]]
-- There are 24 walls for hanging the colour square.
local TWO_ROOMS_VALID_PAINT_LOCATION = 24

local EXPLORE_TEXT_MAP_DICT = {
    PASSIVE = {
        map = SHORT_STRAIGHT_ROOM,
        targetPic = {row=4, col=2, dir='S'},
    },
    TWO_ROOMS = {
        map = TWO_ROOMS,
        targetPic = {row=0, col=0, dir='S'},
    },
    KEY_TO_COLOR = {
        map = SHORT_STRAIGHT_ROOM_WITH_DOOR,
        targetPic = {row=4, col=2, dir='S'},
    }
}

--[[
Setup image room maze.

Example 1:
numImages = 2
imageRoomHeight = 3

*****
**P**
*   *
*   *
*T T*
*****
 *t*
 ***

Example 2:
numImages = 4
imageRoomHeight = 4

*********
****P****
*       *
*       *
*       *
*T T T T*
*********
   *t*
   ***
--]]
local function createImageMaze(opts)
  local numImages = opts.numImages
  local imageRoomHeight = opts.imageRoomHeight or 3
  local centerWidth = 1 + numImages

  local width = 2 * numImages + 1
  -- Set the height to imageRoomHeight + 3 for image room, 2 for finish area.
  local height = (imageRoomHeight + 3) + 2

  -- Initialize the maze. All cells start as '*' (wall).
  local maze = maze_generation:mazeGeneration{
      width = width,
      height = height,
  }
  maze:setEntityCell(2, centerWidth, 'P') -- Avatar start location.

  -- Fill image room with '.' (empty space).
  local imageRoomHeightStart = 3
  local imageRoomHeightEnd = imageRoomHeightStart + imageRoomHeight - 1
  for i = imageRoomHeightStart, imageRoomHeightEnd do
    for j = 2, width - 1 do
      maze:setEntityCell(i, j, '.')
    end
  end

  -- Teleports in final row of image room.
  for n = 1, numImages do
    maze:setEntityCell(imageRoomHeightEnd, 2 * n, 'T')
  end
  -- Teleport target in finish box after hallway.
  maze:setEntityCell(imageRoomHeightEnd + 2, centerWidth, 't')

  print('Generated image maze with entity layer:')
  print(maze:entityLayer())
  io.flush()

  return maze
end

local function createSecondOrderExploreMaze(opts)
  -- An open layout room of size = SECOND_ORDER_EXPLORE_ROOM_SIZE
  -- the avatar is always at top-left corner, while the key is at other random
  -- location. For example, a 3x3 room may be like this:
  -- xxxxx
  -- xPxxx
  -- xKKKx
  -- xKKKx
  -- xxxxx

  roomHeight = opts.roomSize[1]
  roomWidth = opts.roomSize[2]
  local maze = maze_generation:mazeGeneration{
      height = roomHeight + 2,  -- +2 for the two side of walls
      width = roomWidth + 2
  }

  -- Fill image room with 'K' (possible key locations).
  for i = 3, roomHeight + 1 do
    for j = 2, roomWidth + 1 do
      maze:setEntityCell(i, j, 'K')
    end
  end
  maze:setEntityCell(2, 2, 'P') -- Avatar start at top-left corner.

  print('Generated 2nd order explore maze with entity layer:')
  print(maze:entityLayer())
  io.flush()

  return maze
end

local function createDistractorMaze(opts)
    -- Example room with height = 2, width = 3
    -- A are possible apple locations (everywhere)
    -- *****
    -- *APA*
    -- *AAA*
    -- *****

    local roomHeight = opts.roomSize[1]
    local roomWidth = opts.roomSize[2]
    local centerWidth = 1 + math.ceil(roomWidth / 2)
    local maze = maze_generation:mazeGeneration{
        height = roomHeight + 2,  -- +2 for the two side of walls
        width = roomWidth + 2
    }

    -- Fill the room with 'A' for apples. updateSpawnVars decides which to use.
    for i = 2, roomHeight + 1 do
      for j = 2, roomWidth + 1 do
        maze:setEntityCell(i, j, 'A')
      end
    end
    -- Override one cell with 'P' for spawn point.
    maze:setEntityCell(2, centerWidth, 'P')

    print('Generated distractor maze with entity layer:')
    print(maze:entityLayer())
    io.flush()
    return maze
end

local factory = {}
game:console('cg_drawScriptRectanglesAlways 1')

function factory.createLevelApi(kwargs)

  kwargs.episodeLengthSeconds = kwargs.episodeLengthSeconds or
                                DEFAULTS.EPISODE_LENGTH_SECONDS
  kwargs.secondOrderExploreLengthSeconds =
      kwargs.secondOrderExploreLengthSeconds or
      DEFAULTS.SECOND_ORDER_EXPLORE_LENGTH_SECONDS
  kwargs.secondOrderExploreRoomSize = kwargs.secondOrderExploreRoomSize or
                                      DEFAULTS.SECOND_ORDER_EXPLORE_ROOM_SIZE

  kwargs.exploreLengthSeconds = kwargs.exploreLengthSeconds or
                                DEFAULTS.EXPLORE_LENGTH_SECONDS

  kwargs.preExploreDistractorLengthSeconds =
      kwargs.preExploreDistractorLengthSeconds or
      DEFAULTS.PRE_EXPLORE_DISTRACTOR_LENGTH_SECONDS

  kwargs.distractorLengthSeconds = kwargs.distractorLengthSeconds or
                                   DEFAULTS.DISTRACTOR_LENGTH_SECONDS

  kwargs.numImages = kwargs.numImages or DEFAULTS.NUM_IMAGES
  kwargs.correctReward = kwargs.correctReward or DEFAULTS.CORRECT_REWARD
  kwargs.incorrectReward = kwargs.incorrectReward or DEFAULTS.INCORRECT_REWARD

  kwargs.appleReward = kwargs.appleReward or DEFAULTS.APPLE_REWARD
  kwargs.appleRewardProb = kwargs.appleRewardProb or DEFAULTS.APPLE_REWARD_PROB
  kwargs.appleExtraRewardRange =
      kwargs.appleExtraRewardRange or DEFAULTS.APPLE_EXTRA_REWARD_RANGE

  kwargs.imageScale = kwargs.imageScale or DEFAULTS.IMAGE_SCALE
  kwargs.imageRoomHeight = kwargs.imageRoomHeight or DEFAULTS.IMAGE_ROOM_HEIGHT
  kwargs.distractorRoomSize = kwargs.distractorRoomSize or
                              DEFAULTS.DISTRACTOR_ROOM_SIZE
  kwargs.probAppleInDistractorMap = kwargs.probAppleInDistractorMap or
                                    DEFAULTS.PROB_APPLE_IN_DISTRACTOR_MAP
  kwargs.differentDistractRoomTexture = kwargs.differentDistractRoomTexture or
                                        DEFAULTS.DIFFERENT_DISTRACT_ROOM_TEXTURE
  kwargs.differentRewardRoomTexture = kwargs.differentRewardRoomTexture or
                                      DEFAULTS.DIFFERENT_REWARD_ROOM_TEXTURE
  kwargs.differentSecondOrderRoomTexture =
      kwargs.differentSecondOrderRoomTexture or
      DEFAULTS.DIFFERENT_SECOND_ORDER_ROOM_TEXTURE

  kwargs.showKeyColorSquareSeconds = kwargs.showKeyColorSquareSeconds or
                                     DEFAULTS.SHOW_KEY_COLOR_SQUARE_SECONDS

  assert(kwargs.numImages % 2 == 0,
         'numImages must be an even number if there is space between images.')
  assert(kwargs.numImages <= #COLORS,
         'numImages must be <=' .. #COLORS .. ' for simple color images.')

  kwargs.exploreMapMode = kwargs.exploreMapMode or DEFAULTS.EXPLORE_MAP_MODE

  local api = {}

  function api:init(params)
    self._isKeyToPaintingLevel = kwargs.exploreMapMode == 'KEY_TO_COLOR'

    self:_createExploreMap()
    self:_createDistractorMap()
    self:_createImageMap()
    if self._isKeyToPaintingLevel then
      self:_createSecondOrderKeyExploreMap()
    end

    self._imageOrder = {}
    for i = 1, kwargs.numImages do
      self._imageOrder[i] = i
    end
  end

  function api:_createSecondOrderKeyExploreMap()
    -- Create maze to be converted into map.
    local maze = createSecondOrderExploreMaze{
        roomSize = kwargs.secondOrderExploreRoomSize
    }

    -- Create map theme with no wall decals.
    local texture = textureSet
    if kwargs.differentSecondOrderRoomTexture then
      texture = fourthTextureSet
    end

    local keyExploreMapTheme = themes.fromTextureSet{
        textureSet = texture,
        decalFrequency = 0.0,
        floorModelFrequency = 0.0,
    }

    self._secondOrderExploreMap = map_maker:mapFromTextLevel{
        mapName = SECOND_ORDER_EXPLORE_MAP,
        entityLayer = maze:entityLayer(),
        theme = keyExploreMapTheme,
        callback = function (i, j, c, maker)
          local pickup = self:_makePickup(c)
          if pickup then
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:_createExploreMap()
    -- Create map theme and override default wall decal placement.
    local exploreMapTheme = themes.fromTextureSet{
        textureSet = textureSet,
        decalFrequency = 1.0,
        floorModelFrequency = 0.0,
    }

    local exploreMapInfo = EXPLORE_TEXT_MAP_DICT[kwargs.exploreMapMode]
    local targetPic = exploreMapInfo.targetPic
    local exploreMapEntityLayer = exploreMapInfo.map

    -- Note on decalIndex meaning:
    -- decalIndex = 1 to numImages: the id for painting in the imageRoom
    -- decalIndex = numImages + 1, the target image in exploreRoom

    local function _matchTextureLocation(loc, target)
      if loc.i == target.row and loc.j == target.col and
        loc.direction == target.dir then
        return true
      else
        return false
      end
    end

    function exploreMapTheme:placeWallDecals(allWallLocations)
      local wallDecals = {}
      local numPossiblePaintLocation = 0
      for _, loc in pairs(allWallLocations) do
        local decalIndex = nil
        if kwargs.exploreMapMode ~= 'TWO_ROOMS' then
          if _matchTextureLocation(loc, targetPic) then
            decalIndex = kwargs.numImages + 1
          end
        else
          if loc.i > 2 then
            numPossiblePaintLocation = numPossiblePaintLocation + 1
            decalIndex = numPossiblePaintLocation
          end
        end

        if decalIndex then
          local decal = textureSet.wallDecals[decalIndex]
          local actualDecal = {
              tex = decal.tex .. '_alpha',
              scale = kwargs.imageScale,
          }
          wallDecals[#wallDecals + 1] = {
              index = loc.index,
              decal = actualDecal,
          }
          local fullTextureName = "textures/" .. decal.tex
          decalIndices[fullTextureName] = decalIndex
        end
      end
      return wallDecals
    end

    self._exploreMap = map_maker:mapFromTextLevel{
        mapName = EXPLORE_MAP,
        entityLayer = exploreMapEntityLayer,
        theme = exploreMapTheme,
    }
  end

  function api:_createDistractorMap()

    -- Create a maze to be converted into map.
    local maze = createDistractorMaze{
        roomSize = kwargs.distractorRoomSize,
    }

    -- Create a map theme with no wall decals.
    local texture = textureSet
    if kwargs.differentDistractRoomTexture then
      texture = secondTextureSet
    end
    local mapTheme = themes.fromTextureSet{
        textureSet = texture,
        decalFrequency = 0.0,
        floorModelFrequency = 0.0,
    }

    self._distractorMap = make_map.makeMap{
        mapName = DISTRACTOR_MAP,
        mapEntityLayer = maze:entityLayer(),
        theme = mapTheme,
    }
  end

  function api:_createImageMap()
    -- Create a maze to be converted into map.
    local imageMaze = createImageMaze{
        numImages = kwargs.numImages,
        imageRoomHeight = kwargs.imageRoomHeight,
    }

    local texture = textureSet
    if kwargs.differentRewardRoomTexture then
      texture = thirdTextureSet
    end
    -- Create map theme and override default wall decal placement.
    local imageMapTheme = themes.fromTextureSet{
        textureSet = texture,
        decalFrequency = 1.0,
        floorModelFrequency = 0.0,
    }
    local paintingsRow = kwargs.imageRoomHeight + 2
    function imageMapTheme:placeWallDecals(allWallLocations)
      local wallDecals = {}
      local decalCount = 1
      for _, loc in pairs(allWallLocations) do
        if loc.direction == "S" then
          local decalIndex = nil
          if loc.i == paintingsRow then
            -- Only use even columns for paintings.
            if loc.j % 2 == 0 then
              decalIndex = decalCount  -- Will be between 1 and numImages.
              decalCount = decalCount + 1
            end
          end
          if decalIndex then
            local decal = textureSet.wallDecals[decalIndex]
            decal.scale = kwargs.imageScale
            wallDecals[#wallDecals + 1] = {
                index = loc.index,
                decal = decal,
            }
            local fullTextureName = "textures/" .. decal.tex
            decalIndices[fullTextureName] = decalIndex
          end
        end
      end
      return wallDecals
    end

    self._imageMap = map_maker:mapFromTextLevel{
        mapName = IMAGE_MAP,
        entityLayer = imageMaze:entityLayer(),
        theme = imageMapTheme,
        callback = function (i, j, c, maker)
          if c == 'T' then
            return custom_entities.makeTeleporter(
                {imageMaze:toWorldPos(i + 1, j + 1)},
                'teleporter')
          end
          if c == 't' then
            return custom_entities.makeTeleporterTarget(
                {imageMaze:toWorldPos(i + 1, j + 1)},
                'teleporter')
          end
        end
    }
  end

  function api:_prepareKey(keyColor)
    self._holdingKey = false
    local keyInfo = {shape='key', pattern='solid',
                     color1 = {0, 0, 0}, color2={0, 0, 0}}
    self._keyCueColorAlpha = {0, 0, 0, 1}
    self._keyObject = hrp.create(keyInfo)
  end

  function api:start(episode, seed)
    random:seed(seed)
    self._map = nil
    self._time = 0
    self._targetIndex = 1
    self._images = {}
    self._preExploreDistractorLen = kwargs.preExploreDistractorLengthSeconds
    self._distractorLen = kwargs.distractorLengthSeconds

    local colorIndices = {}
    for i = 1, #COLORS do
      colorIndices[i] = i
    end
    random:shuffleInPlace(colorIndices)
    for i = 1, kwargs.numImages do
      local rgb = COLORS[colorIndices[i]]
      self._images[i] = image_utils:createByteImage(3, 3, rgb)
    end

    if kwargs.exploreMapMode == 'TWO_ROOMS' then
      self._images[kwargs.numImages + 1] =
          image_utils:createTransparentImage(3, 3)
      local nPaintPos = TWO_ROOMS_VALID_PAINT_LOCATION
      self._targetPaintLocation = random:uniformInt(1, nPaintPos)
    end

    if self._isKeyToPaintingLevel then
      self:_prepareKey()
      -- Randomly sample the key location in secondOrderExploreMaze
      local nPossibleKeyLocation = (kwargs.secondOrderExploreRoomSize[1] - 1) *
                                   kwargs.secondOrderExploreRoomSize[2]
      self._keyPosition = random:uniformInt(1, nPossibleKeyLocation)
    end

    -- Set instruction channel output to 0. (to indicate final phase reward.)
    self.setInstruction(tostring(0))
  end

  function api:filledRectangles(args)
    if self._map == SECOND_ORDER_EXPLORE_MAP and self._showKeyCue then
      return {{
          x = 12,
          y = 12,
          width = 600,
          height = 200,
          rgba = self._keyCueColorAlpha
      }}
    end
    return {}
  end

  function api:nextMap()
    -- 1. Decide what is the next map.
    if self._map == nil or self._map == IMAGE_MAP then
      if self._isKeyToPaintingLevel then
        self._map = SECOND_ORDER_EXPLORE_MAP
      else
        if self._preExploreDistractorLen > 0.0 then
          self._notExploreYet = true
          self._map = DISTRACTOR_MAP
        else
          self._map = EXPLORE_MAP
        end
      end
    elseif self._map == SECOND_ORDER_EXPLORE_MAP then
      self._notExploreYet = true
      self._map = DISTRACTOR_MAP
    elseif self._map == DISTRACTOR_MAP then
      if self._notExploreYet then
        self._notExploreYet = false
        self._map = EXPLORE_MAP
      else
        self._map = IMAGE_MAP
      end
    elseif self._map == EXPLORE_MAP then
      if self._distractorLen > 0.0 then
        self._map = DISTRACTOR_MAP
      else
        self._map = IMAGE_MAP
      end
    end

    -- 2. Set up properly for the up-coming map.
    if self._map == DISTRACTOR_MAP and self._notExploreYet then
      self._timeOut = self._time + self._preExploreDistractorLen
    elseif self._map == SECOND_ORDER_EXPLORE_MAP then
      self._holdingKey = false
      self._timeOut = self._time + kwargs.secondOrderExploreLengthSeconds
      self._possibleKeyPosCount = 0
    elseif self._map == EXPLORE_MAP then
      self._timeOut = self._time + kwargs.exploreLengthSeconds
    elseif self._map == DISTRACTOR_MAP and not self._notExploreYet then
      self._timeOut = self._time + self._distractorLen
    elseif self._map == IMAGE_MAP then
      self._timeOut = nil
      self._teleportId = 0
      random:shuffleInPlace(self._imageOrder)
      for i, shuffled_i in ipairs(self._imageOrder) do
        if self._targetIndex == shuffled_i then
          self._shuffledTargetIndex = i
        end
      end
    end

    return self._map
  end

  function api:replaceShader(textureName)
    local index = decalIndices[textureName]
    if index then
      textureName = textureName .. '_alpha'
    end
    return textureName
  end

  function api:loadTexture(textureName)
    local fullTextureName = textureName .. "_nonsolid"
    local index = decalIndices[fullTextureName]

    if index then
      if self._map == EXPLORE_MAP and
        kwargs.exploreMapMode == 'TWO_ROOMS' then
        if index == self._targetPaintLocation then
          return self._images[self._targetIndex]  -- Set to arget color.
        else
          return self._images[kwargs.numImages + 1]  -- Set to transparent.
        end
      end

      if index <= kwargs.numImages then
        local shuffledIndex = self._imageOrder[index]
        return self._images[shuffledIndex]
      elseif index == kwargs.numImages + 1 then
        return self._images[self._targetIndex]
      end
    end
  end

 -- PICKUP functions ----------------------------------------------------------

  function api:_makePickup(c)
    if c == 'K' then
      return 'key'
    end
  end

  function api:pickup(spawnId)
    if spawnId == KEY_OBJECT_SPAWN_ID then
      self._holdingKey = true
      self._holdingKeyTime = self._time
      self._showKeyCue = true
    end

    if spawnId == APPLE_ID then
      if kwargs.appleRewardProb >= 1 or
         random:uniformReal(0, 1) < kwargs.appleRewardProb then
        -- the -1 is for the default 1 point for apple in dmlab
        appleReward = kwargs.appleReward +
            random:uniformInt(0, kwargs.appleExtraRewardRange) - 1
        game:addScore(appleReward)
      else
        -- the -1 is to compensate the default 1 point for apple in dmlab
        game:addScore(-1)
      end
    end
  end

  -- TRIGGER functions ---------------------------------------------------------

  function api:canTrigger(teleportId, targetName)
    if string.sub(targetName, 1, 4) == 'door' and not self._holdingKey then
      return false
    end
    return true
  end

  function api:trigger(teleportId, targetName)
    if string.sub(targetName, 1, 4) == 'door' then
      return
    end

    -- Decide if the correct teleport is triggered.
    local reward = 0
    if teleportId == self._shuffledTargetIndex then
      self.setInstruction(tostring(kwargs.correctReward))
      reward = kwargs.correctReward
    else
      self.setInstruction(tostring(kwargs.incorrectReward))
      reward = kwargs.incorrectReward
    end

    game:addScore(reward)
    self._timeOut = self._time + 0.2
  end

  function api:hasEpisodeFinished(timeSeconds)
    self._time = timeSeconds

    -- Decide the timing of showing the key cue.
    if self._isKeyToPaintingLevel and self._holdingKey then
      showTime = self._time - self._holdingKeyTime
      if showTime > kwargs.showKeyColorSquareSeconds then
        self._showKeyCue = false
      end
    end

    if self._map == EXPLORE_MAP or self._map == DISTRACTOR_MAP or
       self._map == SECOND_ORDER_EXPLORE_MAP then
      if timeSeconds > self._timeOut then
        game:finishMap()
      end
      return false
    else  -- In the image room map, timeout only after been teleported.
      return self._timeOut and timeSeconds > self._timeOut
    end
  end

  -- END TRIGGER functions -----------------------------------------------------

  function api:updateSpawnVars(spawnVars)
    local classname = spawnVars.classname
    if classname == "info_player_start" then
      -- Spawn facing South.
      spawnVars.angle = "-90"
      spawnVars.randomAngleRange = "0"
    elseif classname == "trigger_teleport" then
      self._teleportId = self._teleportId + 1
      spawnVars.id = tostring(self._teleportId)
    elseif classname == "func_door" then
      spawnVars.id = tostring(DOOR_ID)
      spawnVars.wait = "1000000" -- Open the door for long time.
    elseif classname == "apple_reward" then
      local useApple = false
      if kwargs.probAppleInDistractorMap > 0 then
        useApple = random:uniformReal(0, 1) < kwargs.probAppleInDistractorMap
        spawnVars.id = tostring(APPLE_ID)
      end
      if not useApple then
        return nil
      end
    elseif classname == "key" then
      self._possibleKeyPosCount = self._possibleKeyPosCount + 1
      if self._keyPosition == self._possibleKeyPosCount then
        spawnVars.id = tostring(KEY_OBJECT_SPAWN_ID)
        spawnVars.classname = self._keyObject
      else
        return nil
      end
    end
    return spawnVars
  end

  custom_observations.decorate(api)
  pickup_decorator.decorate(api)
  setting_overrides.decorate{
      api = api,
      apiParams = kwargs,
      decorateWithTimeout = true
  }
  return api
end

return factory
