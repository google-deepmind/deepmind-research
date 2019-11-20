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
local game = require 'dmlab.system.game'
local maze_generation = require 'dmlab.system.maze_generation'
local pickup_decorator = require 'decorators.human_recognisable_pickups'
local random = require 'common.random'
local setting_overrides = require 'decorators.setting_overrides'
local texture_sets = require 'themes.texture_sets'
local themes = require 'themes.themes'
local hrp = require 'common.human_recognisable_pickups'

local SHOW_COLOR_CUE_SECOND = 0.25
local EPISODE_LENGTH_SECONDS = 30
local EXPLORE_LENGTH_SECONDS = 10
local DISTRACTOR_LENGTH_SECONDS = 10
local NUM_OBJECTS = 3
local PROB_GOOD_OBJECT = 0.5
local GAURANTEE_GOOD_OBJECTS = 0
local GAURANTEE_BAD_OBJECTS = 0

local PROB_APPLE_IN_DISTRACTOR_MAP = 0.3
local APPLE_REWARD = 5
local APPLE_EXTRA_REWARD_RANGE = 0
local DISTRACTOR_ROOM_SIZE = {11, 11}
local APPLE_ID = 1000
local CORRECT_REWARD = 2
local INCORRECT_REWARD = -1
local ROOM_SIZE = {3, 5}
local OBJECT_SCALE = 1.62

local EXPLORE_MAP = "exploreMap"
local DISTRACTOR_MAP = "distractorMap"
local EXPLOIT_MAP = "exploitMap"


local DIFFERENT_DISTRACT_ROOM_TEXTURE = false

-- Set texture set for all maps.
local textureSet = texture_sets.TRON
local secondTextureSet = texture_sets.TETRIS

-- Takes goal/location:i -> i
local function nameToLocationId(name)
  return tonumber(name:match('^.+:(%d+)$'))
end

-- Takes goal/location:i -> goal/pickup
local function nameToLocationClass(name)
  return name:match('^(.+):%d+$')
end

local factory = {}
game:console('cg_drawScriptRectanglesAlways 1')

function factory.createLevelApi(kwargs)
  kwargs.episodeLengthSeconds = kwargs.episodeLengthSeconds or
                                EPISODE_LENGTH_SECONDS
  kwargs.exploreLengthSeconds = kwargs.exploreLengthSeconds or
                                EXPLORE_LENGTH_SECONDS
  if kwargs.distractorLengthSeconds == 0 then
    kwargs.skipDistractor = true
  else
    kwargs.distractorLengthSeconds = kwargs.distractorLengthSeconds or
                                     DISTRACTOR_LENGTH_SECONDS
  end
  kwargs.numObjects = kwargs.numObjects or NUM_OBJECTS
  kwargs.probGoodObject = kwargs.probGoodObject or PROB_GOOD_OBJECT
  kwargs.guaranteeGoodObjects = kwargs.guaranteeGoodObjects or
                                GAURANTEE_GOOD_OBJECTS
  kwargs.guaranteeBadObjects = kwargs.guaranteeBadObjects or
                               GAURANTEE_BAD_OBJECTS
  kwargs.correctReward = kwargs.correctReward or CORRECT_REWARD
  kwargs.incorrectReward = kwargs.incorrectReward or INCORRECT_REWARD
  kwargs.roomSize = kwargs.roomSize or ROOM_SIZE
  kwargs.distractorRoomSize = kwargs.distractorRoomSize or DISTRACTOR_ROOM_SIZE
  kwargs.probAppleInDistractorMap = kwargs.probAppleInDistractorMap or
                                    PROB_APPLE_IN_DISTRACTOR_MAP
  kwargs.differentDistractRoomTexture = kwargs.differentDistractRoomTexture or
                                        DIFFERENT_DISTRACT_ROOM_TEXTURE
  kwargs.appleReward = kwargs.appleReward or APPLE_REWARD
  kwargs.appleExtraRewardRange = kwargs.appleExtraRewardRange or
                                 APPLE_EXTRA_REWARD_RANGE
  kwargs.objectScale = kwargs.objectScale or OBJECT_SCALE

  local api = {}

  function api:init(params)
    self:_createExploreMap()
    self:_createDistractorMap()
    self:_createExploitMap()
  end

  function api:pickup(spawnId)
    if self._map == EXPLORE_MAP then
      -- Setup to show color cue.
      self._showObjectCue = true
      self._cueColor = self._objects[spawnId].cueColor
      self._cueStartTime = self._time
    elseif self._map == EXPLOIT_MAP then
      -- Give corresponding reward and termiante when all good objects collected
      game:addScore(self._objects[spawnId].reward)
      -- Update the instruction channel (to record final phase rewards.)
      self._finalRewardMainTask = (
          self._finalRewardMainTask  + self._objects[spawnId].reward)
      self.setInstruction(tostring(self._finalRewardMainTask))
    end

    if spawnId == APPLE_ID then
      -- note the -1 to offset default 1 point for apple in dmlab
      appleReward = kwargs.appleReward +
          random:uniformInt(0, kwargs.appleExtraRewardRange) - 1
      game:addScore(appleReward)
    end
  end

  function api:_createRoomCommon()
    local roomHeight = kwargs.roomSize[1]
    local roomWidth = kwargs.roomSize[2]
    local maze = maze_generation:mazeGeneration{
        height = roomHeight + 2,
        width = roomWidth + 2
    }

    -- Set (2,2) as 'P' for the avatar location.
    -- Set (i,j) as 'O' for possible object location if i%2 == 0 && j%2 == 0.
    -- Otherwise, fill with '.' for empty location.
    self._numLocations = 0
    for i = 2, roomHeight + 1 do
      for j = 2, roomWidth + 1 do
        if i == 2 and j == 2 then
          maze:setEntityCell(i, j, 'P')
        elseif i % 2 == 0 and j % 2 == 0 then
          maze:setEntityCell(i, j, 'O')
          self._numLocations = self._numLocations + 1
        else
          maze:setEntityCell(i, j, '.')
        end
      end
    end

    return maze
  end

  function api:_createExploreMap()
    maze = self:_createRoomCommon()
    print('Generated explore maze with entity layer:')
    print(maze:entityLayer())
    io.flush()

    local mapTheme = themes.fromTextureSet{
        textureSet = textureSet,
        decalFrequency = 0.0,
    }

    local counter = 1
    self._exploreMap = make_map.makeMap{
        mapName = EXPLORE_MAP,
        mapEntityLayer = maze:entityLayer(),
        theme = mapTheme,
        callback = function (i, j, c, maker)
          if c == 'O' then
            pickup = 'location:' .. counter
            counter = counter + 1
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:_createDistractorMap()
    -- Create map theme with no wall decals.
    local distractorMapTheme = themes.fromTextureSet{
        textureSet = textureSet,
        decalFrequency = 0.0,
    }

    -- Example room with height = 2, width = 3
    -- *****
    -- *APA*
    -- *AAA*
    -- *****
    local roomHeight = kwargs.distractorRoomSize[1]
    local roomWidth = kwargs.distractorRoomSize[2]
    centerWidth = 1 + math.ceil(roomWidth / 2)
    local maze = maze_generation:mazeGeneration{
        height = roomHeight + 2,
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

    local texture = textureSet
    if kwargs.differentDistractRoomTexture then
      texture = secondTextureSet
    end
    local mapTheme = themes.fromTextureSet{
        textureSet = texture,
        decalFrequency = 0.0,
    }
    self._distractMap = make_map.makeMap{
        mapName = DISTRACTOR_MAP,
        mapEntityLayer = maze:entityLayer(),
        theme = mapTheme,
    }
  end

  function api:_createExploitMap()
    maze = self:_createRoomCommon()
    print('Generated exploit maze with entity layer:')
    print(maze:entityLayer())
    io.flush()

    local mapTheme = themes.fromTextureSet{
        textureSet = textureSet,
        decalFrequency = 0.0,
    }

    local counter = 1
    self.exploitMap = make_map.makeMap{
        mapName = EXPLOIT_MAP,
        mapEntityLayer = maze:entityLayer(),
        theme = mapTheme,
        useSkybox = false,
        callback = function (i, j, c, maker)
          if c == 'O' then
            pickup = 'location:' .. counter
            counter = counter + 1
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:_generateRandomObjects()
    -- 1. Generate a random list of positive/negative reward, `objectValence`
    -- as function(numObjects, guaranteeGood, guaranteeBad, probGoodObject)

    local objectValence = {}
    for i = 1, kwargs.numObjects do
      if i <= kwargs.guaranteeGoodObjects then
        objectValence[i] = 1
      elseif i<= kwargs.guaranteeGoodObjects + kwargs.guaranteeBadObjects then
        objectValence[i] = -1
      else
        if random:uniformReal(0, 1) < kwargs.probGoodObject then
          objectValence[i] = 1
        else
          objectValence[i] = -1
        end
      end
    end
    random:shuffleInPlace(objectValence)

    -- 2. Generate random objects and link to the object valence above.
    local objects = hrp.uniquelyShapedPickups(kwargs.numObjects)
    for i = 1, kwargs.numObjects do
      objects[i].scale= kwargs.objectScale
    end

    self._objects = {}
    for i, object in ipairs(objects) do
      self._objects[i] = {}
      self._objects[i].data = hrp.create(object)
      if objectValence[i] == 1 then
        self._objects[i].isGoodObject = true
        self._objects[i].reward = kwargs.correctReward
        self._objects[i].cueColor = {0, 1, 0, 1} -- green means good
      else
        self._objects[i].isGoodObject = false
        self._objects[i].reward = kwargs.incorrectReward
        self._objects[i].cueColor = {1, 0, 0, 1} -- red means bad
      end
    end
  end

  function api:start(episode, seed)
    random:seed(seed)

    -- Setup a random mapping from locationId to pickupId
    -- There should be more locationId than pickupId
    -- The location set with pickupId == 0 will have no object presented there.
    self._mapLocationIdToPickupId = {}
    for i = 1, self._numLocations do
      if i <= kwargs.numObjects then
        self._mapLocationIdToPickupId[i] = i
      else
        self._mapLocationIdToPickupId[i] = 0
      end
    end
    random:shuffleInPlace(self._mapLocationIdToPickupId)

    self:_generateRandomObjects()
    self._map = nil
    self._numTrials = 0
    self._timeOut = kwargs.exploreLengthSeconds

    -- Set the instruction channel to record the rewards in the final phase.
    self._finalRewardMainTask = 0
    self.setInstruction("0")
  end

  function api:nextMap()
    if self._map == nil then  -- Start of episode.
      self._map = EXPLORE_MAP
    elseif not kwargs.skipDistractor and self._map == EXPLORE_MAP then
      -- Move from explore to distractor.
      self._map = DISTRACTOR_MAP
      self._timeOut = self._time + kwargs.distractorLengthSeconds
    elseif (kwargs.skipDistractor and self._map == EXPLORE_MAP)
           or self._map == DISTRACTOR_MAP then
      -- Move from distractor or explore map to exploit map.
      self._map = EXPLOIT_MAP
      random:shuffleInPlace(self._mapLocationIdToPickupId)
      self._timeOut = nil
    end

    return self._map
  end

  function api:hasEpisodeFinished(timeSeconds)
    self._time = timeSeconds
    if self._showObjectCue then
      if self._time - self._cueStartTime > SHOW_COLOR_CUE_SECOND then
        self._showObjectCue = false
      end
    end

    if self._map == EXPLORE_MAP or self._map == DISTRACTOR_MAP then
      if timeSeconds > self._timeOut then
        game:finishMap()
      end
      return false
    end
  end

  -- END TRIGGER functions -----------------------------------------------------
  function api:filledRectangles(args)
    if self._map == EXPLORE_MAP and self._showObjectCue then
      return {{
          x = 12,
          y = 12,
          width = 600,
          height = 300,
          rgba = self._cueColor,
      }}
    end
    return {}
  end

  function api:updateSpawnVars(spawnVars)
    local classname = spawnVars.classname
    if classname == "info_player_start" then
      -- Spawn facing South.
      spawnVars.angle = "-90"
      spawnVars.randomAngleRange = "0"
    elseif classname == "apple_reward" then
      local useApple = false
      if kwargs.probAppleInDistractorMap > 0 then
        useApple = random:uniformReal(0, 1) < kwargs.probAppleInDistractorMap
        spawnVars.id = tostring(APPLE_ID)
      end
      if not useApple then
        return nil
      end
    else
      -- Allocate objects onto the map by mapLocationIdToPickupId.
      local locationClass = nameToLocationClass(classname)
      if locationClass then
        local locationId = nameToLocationId(classname)
        id = self._mapLocationIdToPickupId[locationId]
        if id == 0 then
          return nil
        else
          spawnVars.classname = self._objects[id].data
          spawnVars.id = tostring(id)
        end
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
